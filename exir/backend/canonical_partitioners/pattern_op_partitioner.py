# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import collections
import itertools
from copy import copy
from typing import Dict, List, Optional, Sequence, Set

import torch
# from torch.fx.passes.infra.partitioner import CapabilityBasedPartitioner, Partition
from torch.fx.passes.infra.partitioner import Partition
from torch.fx.passes.operator_support import any_chain, OperatorSupportBase
from torch.fx.passes.utils.matcher_utils import SubgraphMatcher
from torch.fx.graph_module import GraphModule
from torch.fx.node import Node, _get_qualified_name
from torch.fx.passes.utils.fuser_utils import fuse_by_partitions

# def partition_nodes(
#     graph_module: torch.fx.GraphModule,
# ) -> List[Partition]:
#     return []

TOTAL_NUM_NODES = 819

NUM_NODES_PER_SLICE = 800

class _DependencyViewer:
    def __init__(self, graph_module: GraphModule):
        self.upstreams = collections.defaultdict(set)
        self.downstreams = collections.defaultdict(set)

        for node in graph_module.graph.nodes:
            for input_node in node.all_input_nodes:
                # add input_node and input_node's upstream dependency
                self.upstreams[node].add(input_node)
                self.upstreams[node].update(self.upstreams[input_node])

        for node in reversed(graph_module.graph.nodes):
            for output_node in node.users:
                # add output_node and output_node's downstream dependency
                self.downstreams[node].add(output_node)
                self.downstreams[node].update(self.downstreams[output_node])

    def downstreams_of(self, node: Node) -> Set[Node]:
        return self.downstreams[node]

    def upstreams_of(self, node: Node) -> Set[Node]:
        return self.upstreams[node]

class CapabilityPartitioner:

    def __init__(self,
                 graph_module: GraphModule,
                 operator_support: OperatorSupportBase,
                 allows_single_node_partition: bool = False,
                 non_compute_ops: Optional[Sequence[str]] = None,
                 allowed_single_node_partition_ops: Optional[Sequence[str]] = None,
                 ) -> None:
        self.graph_module = graph_module
        self.operator_support = operator_support
        self.allows_single_node_partition = allows_single_node_partition
        self.non_compute_ops = non_compute_ops if non_compute_ops is not None else []
        self.allowed_single_node_partition_ops = (
            allowed_single_node_partition_ops
            if allowed_single_node_partition_ops is not None
            else []
        )
        self.dependency_viewer = _DependencyViewer(graph_module)

    def __is_node_supported(self, node: Node) -> bool:
        return (
            self.operator_support.is_node_supported(dict(self.graph_module.named_modules()), node)
        )

    def propose_sliced_partitions(self) -> List[Partition]:
        """
        Propose partition by merging each new node into the existing partition.
        """
        print("DX propose_sliced_partitions")

        # partition_map is a mapping from partition id to a set of partition id's.
        # The value set contains all the partition ids that can be reached by doing a
        # DFS starting from the partition id in the key.
        partition_map : Dict[int, Set] = collections.defaultdict(set)

        # assumptions: nodes in candidate list is sorted in topological order
        assignment: Dict[Node, int] = {}   # mapping from node to partition_id
        partitions_by_id: Dict[int, Partition] = {}  # mapping from partition_id to partition
        new_partition_id = itertools.count()

        # try to merge partition other_id into partition self_id
        # merge only happens if the end graph doesn't contain cyclic dependency
        # returns `True` when merge happens, `False` otherwise.
        def maybe_merge_partition(self_id: int, other_id: int):
            """
            Check feasibility and merge 2 partitions; delete the old partition from partition_map.
            """
            # Stop merging at last partition
            self_num_nodes = len(partitions_by_id[self_id].nodes)
            others_num_nodes = len(partitions_by_id[other_id].nodes)
            if self_num_nodes > others_num_nodes:
                return maybe_merge_partition(other_id, self_id)
            print(f"DX maybe_merge_partition other_id: {other_id}, self_id: {self_id}")
            # if self_num_nodes == 1 or others_num_nodes > NUM_NODES_PER_SLICE:

            #     return False
            # if others_num_nodes > NUM_NODES_PER_SLICE or self_num_nodes > NUM_NODES_PER_SLICE:
            #     print(f"DX: partition too large exiting, partitin[{other_id}] num_nodes={others_num_nodes}; partitin[{self_id}] num_nodes={self_num_nodes}")
            #     return False

            # merged_nodes is the union of nodes in two partition to-be-merged
            merged_nodes = copy(partitions_by_id[self_id].nodes)
            merged_nodes.update(partitions_by_id[other_id].nodes)

            def dfs_iter_find_cycle(all_user_nodes: List[Node]):
                for user_node in all_user_nodes:
                    visited_partition_ids = set()

                    for path_node in self.dependency_viewer.downstreams_of(user_node):
                        # If any of the nodes in the dfs path of this node are in the merged_nodes
                        # list then there is a cycle in the graph.
                        if path_node in merged_nodes:
                            return True

                        # If any of the nodes in the dfs path of this node are in the assignment
                        # map then we have to make sure that the partitions that these nodes belong
                        # to do not form a cycle with the current partitions being merged. This means
                        # iterating through all the nodes in all the parititons that are traversed in
                        # the dfs path and checking if they are in the merged_nodes list.
                        if path_node in assignment:
                            partition_id = assignment[path_node]
                            # If the partition id has already been visited then we know that it doesn't
                            # form a cycle with the current partitions being merged.
                            if partition_id in visited_partition_ids:
                                continue
                            p_map = partition_map[partition_id]
                            if self_id in p_map or other_id in p_map:
                                return True

                            visited_partition_ids.add(partition_id)

                return False

            # check if merge would create cyclic dependency.
            all_user_nodes = []
            for node in merged_nodes:
                for user_node in node.users:
                    if user_node not in merged_nodes:
                        all_user_nodes.append(user_node)

            if dfs_iter_find_cycle(all_user_nodes):
                # return false indicating cyclic dependency found and
                # merge is aborted
                return False

            # no cyclic dependency found, move forward with the merge
            # updating partition nodes
            partitions_by_id[self_id].nodes = merged_nodes
            # updating assignment map
            for node in partitions_by_id[other_id].nodes:
                assignment[node] = self_id
            # delete other partition
            del partitions_by_id[other_id]

            partition_map[self_id] = partition_map[self_id].union(partition_map[other_id])
            del partition_map[other_id]

            return True

        def merge_single_node(node: Node, id: Optional[int]):
            print("DX merge_single_node")
            def _update_partition_map(node: Node, id: int):
                # Iterate through all the downstream nodes of this node and update the partition map
                # to indicate that there is a path from the partition id of this node to the target
                # partition id.
                downstream_nodes = self.dependency_viewer.downstreams_of(node)
                for curr_node in downstream_nodes:
                    target_id = assignment.get(curr_node, None)
                    if target_id is not None:
                        partition_map[id].add(target_id)

                # Iterate through all the upstream nodes of this node and update the partition map
                # to indicate that there is a path from the partition id of the upstream node to the
                # current node's partition id.
                upstream_nodes = self.dependency_viewer.upstreams_of(node)
                for curr_node in upstream_nodes:
                    source_id = assignment.get(curr_node, None)
                    if source_id is not None:
                        partition_map[source_id].add(id)

            if node in assignment:
                partitions_by_id[assignment[node]].remove_node(node)

            if id is None:
                assignment.pop(node)
            elif id not in partitions_by_id:
                assignment[node] = id
                partitions_by_id[id] = Partition(id=id, nodes=[node])
                _update_partition_map(node, id)
            else:
                assignment[node] = id
                partitions_by_id[id].add_node(node)
                _update_partition_map(node, id)

        print("Proposing partitions...")
        print(f"DX Current num partitions: {len(partitions_by_id.keys())}")
        for node in reversed(self.graph_module.graph.nodes):
            # use Dict as an ordered set to ensure deterministic partitioning result, don't care value
            merge_candidates: Dict[int, None] = {}

            # Note a limited horizontal fusion is enabled:
            #   when `node` is not supported, the code below attempts to fuse consumer of `node`.
            #
            # I don't see a need to add a knob to disable horizontal fusion yet, we can short-cut
            # the fusion by adding an `else` block here to skip horizontal fusion.
            if self.__is_node_supported(node) and node not in assignment:
                print("DX calling merge_single_node")
                partition_id = next(new_partition_id)
                merge_single_node(node, partition_id)
                merge_candidates[partition_id] = None

            # merge all possible partitions
            for node in assignment:
                merge_candidates[assignment[node]] = None

            # TODO: move the finished partition out of the merge_candidates @nocommit
            merge_candidates_list = list(merge_candidates.keys())
            if len(merge_candidates_list) > 1:
                self_id = merge_candidates_list[0]
                for other_id in merge_candidates_list[1:]:
                    # note: merge partition `other_id` into partition `self_id` if
                    # it doesn't create cyclic dependency in the graph, otherwise,
                    # this is a no-op
                    print(f"DX calling maybe_merge_partition, num partitions: {len(partitions_by_id.keys())}")
                    maybe_merge_partition(self_id, other_id)

        # post processing to re-assign "getitem" nodes into upstream partition
        print("Reassigning getitem nodes to its producer node's partition...")
        nodes_reassignment: Dict[Node, int] = {}
        for node in self.graph_module.graph.nodes:
            is_tuple_output = True
            for user in node.users:
                if user.op != "call_function" or \
                   _get_qualified_name(user.target) != "_operator.getitem":     # type: ignore[arg-type]
                    is_tuple_output = False
                    break

            # node has tuple outputs, re-assign all following getitem node into node's partition
            if is_tuple_output:
                id = assignment.get(node, None)     # type: ignore[arg-type]
                for user in node.users:
                    if assignment.get(user, None) != id:    # type: ignore[arg-type]
                        nodes_reassignment[user] = id  # type: ignore[assignment]
        for node, id in nodes_reassignment.items():
            merge_single_node(node, id)

        # filter out single node partitions
        if not self.allows_single_node_partition:
            print("Filtering out single node partitions...")
            default_non_compute_ops = {"torch.ops.aten.view", "_operator.getitem"}
            non_compute_ops = default_non_compute_ops.union(set(self.non_compute_ops))
            partitions_to_remove: List[int] = []
            for id, partition in partitions_by_id.items():
                compute_node_count = 0
                for node in partition.nodes:
                    if node.op == "call_function":
                        assert callable(node.target)
                        if _get_qualified_name(node.target) not in non_compute_ops:
                            compute_node_count += 1
                        if _get_qualified_name(node.target) in self.allowed_single_node_partition_ops:
                            compute_node_count += 1
                if compute_node_count <= 1:
                    partitions_to_remove.append(id)
            for id in partitions_to_remove:
                del partitions_by_id[id]

        print("Partitions proposed:")
        for id, partition in partitions_by_id.items():
            # print("partition #%s: %s", id, [node.name for node in partition.nodes])
            print(f"partition {id}, num nodes: {len(partition.nodes)}")

        # End of propose_sliced_partitions()
        partition_list = list(partitions_by_id.values())
        return partition_list

    def propose_partitions(self) -> List[Partition]:
        """
        Propose partition by merging each new node into the existing partition.
        """
        print("DX propose_partitions")

        # partition_map is a mapping from partition id to a set of partition id's.
        # The value set contains all the partition ids that can be reached by doing a
        # DFS starting from the partition id in the key.
        partition_map : Dict[int, Set] = collections.defaultdict(set)

        # assumptions: nodes in candidate list is sorted in topological order
        assignment: Dict[Node, int] = {}   # mapping from node to partition_id
        partitions_by_id: Dict[int, Partition] = {}  # mapping from partition_id to partition
        new_partition_id = itertools.count()

        # try to merge partition other_id into partition self_id
        # merge only happens if the end graph doesn't contain cyclic dependency
        # returns `True` when merge happens, `False` otherwise.
        def maybe_merge_partition(self_id: int, other_id: int):
            """
            Check feasibility and merge 2 partitions; delete the old partition from partition_map.
            """
            # Stop merging at last partition
            self_num_nodes = len(partitions_by_id[self_id].nodes)
            others_num_nodes = len(partitions_by_id[other_id].nodes)
            if self_num_nodes > others_num_nodes:
                return maybe_merge_partition(other_id, self_id)
            print(f"DX maybe_merge_partition other_id: {other_id}, self_id: {self_id}")
            # if self_num_nodes == 1 or others_num_nodes > NUM_NODES_PER_SLICE:

            #     return False
            # if others_num_nodes > NUM_NODES_PER_SLICE or self_num_nodes > NUM_NODES_PER_SLICE:
            #     print(f"DX: partition too large exiting, partitin[{other_id}] num_nodes={others_num_nodes}; partitin[{self_id}] num_nodes={self_num_nodes}")
            #     return False

            # merged_nodes is the union of nodes in two partition to-be-merged
            merged_nodes = copy(partitions_by_id[self_id].nodes)
            merged_nodes.update(partitions_by_id[other_id].nodes)

            def dfs_iter_find_cycle(all_user_nodes: List[Node]):
                for user_node in all_user_nodes:
                    visited_partition_ids = set()

                    for path_node in self.dependency_viewer.downstreams_of(user_node):
                        # If any of the nodes in the dfs path of this node are in the merged_nodes
                        # list then there is a cycle in the graph.
                        if path_node in merged_nodes:
                            return True

                        # If any of the nodes in the dfs path of this node are in the assignment
                        # map then we have to make sure that the partitions that these nodes belong
                        # to do not form a cycle with the current partitions being merged. This means
                        # iterating through all the nodes in all the parititons that are traversed in
                        # the dfs path and checking if they are in the merged_nodes list.
                        if path_node in assignment:
                            partition_id = assignment[path_node]
                            # If the partition id has already been visited then we know that it doesn't
                            # form a cycle with the current partitions being merged.
                            if partition_id in visited_partition_ids:
                                continue
                            p_map = partition_map[partition_id]
                            if self_id in p_map or other_id in p_map:
                                return True

                            visited_partition_ids.add(partition_id)

                return False

            # check if merge would create cyclic dependency.
            all_user_nodes = []
            for node in merged_nodes:
                for user_node in node.users:
                    if user_node not in merged_nodes:
                        all_user_nodes.append(user_node)

            if dfs_iter_find_cycle(all_user_nodes):
                # return false indicating cyclic dependency found and
                # merge is aborted
                return False

            # no cyclic dependency found, move forward with the merge
            # updating partition nodes
            partitions_by_id[self_id].nodes = merged_nodes
            # updating assignment map
            for node in partitions_by_id[other_id].nodes:
                assignment[node] = self_id
            # delete other partition
            del partitions_by_id[other_id]

            partition_map[self_id] = partition_map[self_id].union(partition_map[other_id])
            del partition_map[other_id]

            return True

        def merge_single_node(node: Node, id: Optional[int]):
            print("DX merge_single_node")
            def _update_partition_map(node: Node, id: int):
                # Iterate through all the downstream nodes of this node and update the partition map
                # to indicate that there is a path from the partition id of this node to the target
                # partition id.
                downstream_nodes = self.dependency_viewer.downstreams_of(node)
                for curr_node in downstream_nodes:
                    target_id = assignment.get(curr_node, None)
                    if target_id is not None:
                        partition_map[id].add(target_id)

                # Iterate through all the upstream nodes of this node and update the partition map
                # to indicate that there is a path from the partition id of the upstream node to the
                # current node's partition id.
                upstream_nodes = self.dependency_viewer.upstreams_of(node)
                for curr_node in upstream_nodes:
                    source_id = assignment.get(curr_node, None)
                    if source_id is not None:
                        partition_map[source_id].add(id)

            if node in assignment:
                partitions_by_id[assignment[node]].remove_node(node)

            if id is None:
                assignment.pop(node)
            elif id not in partitions_by_id:
                assignment[node] = id
                partitions_by_id[id] = Partition(id=id, nodes=[node])
                _update_partition_map(node, id)
            else:
                assignment[node] = id
                partitions_by_id[id].add_node(node)
                _update_partition_map(node, id)

        print("Proposing partitions...")
        print(f"DX Current num partitions: {len(partitions_by_id.keys())}")
        for node in reversed(self.graph_module.graph.nodes):
            # use Dict as an ordered set to ensure deterministic partitioning result, don't care value
            merge_candidates: Dict[int, None] = {}

            # Note a limited horizontal fusion is enabled:
            #   when `node` is not supported, the code below attempts to fuse consumer of `node`.
            #
            # I don't see a need to add a knob to disable horizontal fusion yet, we can short-cut
            # the fusion by adding an `else` block here to skip horizontal fusion.
            if self.__is_node_supported(node) and node not in assignment:
                print("DX calling merge_single_node")
                partition_id = next(new_partition_id)
                merge_single_node(node, partition_id)
                merge_candidates[partition_id] = None

            # merge all possible partitions
            for node in assignment:
                merge_candidates[assignment[node]] = None

            # TODO: move the finished partition out of the merge_candidates @nocommit
            merge_candidates_list = list(merge_candidates.keys())
            if len(merge_candidates_list) > 1:
                self_id = merge_candidates_list[0]
                for other_id in merge_candidates_list[1:]:
                    # note: merge partition `other_id` into partition `self_id` if
                    # it doesn't create cyclic dependency in the graph, otherwise,
                    # this is a no-op
                    print(f"DX calling maybe_merge_partition, num partitions: {len(partitions_by_id.keys())}")
                    maybe_merge_partition(self_id, other_id)

        # post processing to re-assign "getitem" nodes into upstream partition
        print("Reassigning getitem nodes to its producer node's partition...")
        nodes_reassignment: Dict[Node, int] = {}
        for node in self.graph_module.graph.nodes:
            is_tuple_output = True
            for user in node.users:
                if user.op != "call_function" or \
                   _get_qualified_name(user.target) != "_operator.getitem":     # type: ignore[arg-type]
                    is_tuple_output = False
                    break

            # node has tuple outputs, re-assign all following getitem node into node's partition
            if is_tuple_output:
                id = assignment.get(node, None)     # type: ignore[arg-type]
                for user in node.users:
                    if assignment.get(user, None) != id:    # type: ignore[arg-type]
                        nodes_reassignment[user] = id  # type: ignore[assignment]
        for node, id in nodes_reassignment.items():
            merge_single_node(node, id)

        # filter out single node partitions
        if not self.allows_single_node_partition:
            print("Filtering out single node partitions...")
            default_non_compute_ops = {"torch.ops.aten.view", "_operator.getitem"}
            non_compute_ops = default_non_compute_ops.union(set(self.non_compute_ops))
            partitions_to_remove: List[int] = []
            for id, partition in partitions_by_id.items():
                compute_node_count = 0
                for node in partition.nodes:
                    if node.op == "call_function":
                        assert callable(node.target)
                        if _get_qualified_name(node.target) not in non_compute_ops:
                            compute_node_count += 1
                        if _get_qualified_name(node.target) in self.allowed_single_node_partition_ops:
                            compute_node_count += 1
                if compute_node_count <= 1:
                    partitions_to_remove.append(id)
            for id in partitions_to_remove:
                del partitions_by_id[id]

        print("Partitions proposed:")
        for id, partition in partitions_by_id.items():
            # print("partition #%s: %s", id, [node.name for node in partition.nodes])
            print(f"partition {id}, num nodes: {len(partition.nodes)}")

        # End of propose_partitions()
        return list(partitions_by_id.values())

    def fuse_partitions(self, partitions: List[Partition]) -> GraphModule:
        print("Fusing partitions...")
        # fuse_by_partitions expects partitions in List[List[Node]]: [ [node0, node1], [node2, node3] ]
        return fuse_by_partitions(self.graph_module, [list(partition.nodes) for partition in partitions])

    # remove non-compute-ops that sits at the boundary of a partition.
    def remove_bookend_non_compute_ops(self, partitions: List[Partition]):
        non_compute_ops = set(self.non_compute_ops)

        def is_non_compute_node(node: Node):
            return node.op == "call_function" and \
                _get_qualified_name(node.target) in non_compute_ops  # type: ignore[arg-type]

        # cache transparent nodes
        transparent_input_nodes: Dict[Node, bool] = {}
        transparent_output_nodes: Dict[Node, bool] = {}

        def is_transparent_input_node(node: Node, partition: Set[Node], removed_nodes: Set[Node]):
            if node.op == "placeholder" or (node not in partition) or (node in removed_nodes):
                return True
            if node in transparent_input_nodes:
                return transparent_input_nodes[node]
            if is_non_compute_node(node):
                for input_n in node.all_input_nodes:
                    if not is_transparent_input_node(input_n, partition, removed_nodes):
                        transparent_input_nodes[node] = False
                        return False
                transparent_input_nodes[node] = True
                return True
            transparent_input_nodes[node] = False
            return False

        def is_transparent_output_node(node: Node, partition: Set[Node], removed_nodes: Set[Node]):
            if node.op == "placeholder" or (node not in partition) or (node in removed_nodes):
                return True
            if node in transparent_output_nodes:
                return transparent_output_nodes[node]
            if is_non_compute_node(node):
                for output_n in node.users:
                    if not is_transparent_output_node(output_n, partition, removed_nodes):
                        transparent_output_nodes[node] = False
                        return False
                transparent_output_nodes[node] = True
                return True
            transparent_output_nodes[node] = False
            return False

        for partition in partitions:
            # Note it's ok to use `set` here, since we are only query if a node
            # has been removed. We are NEVER going to iterate on nodes inside
            # the set.
            remove_node: Set[Node] = set()
            for node in partition.nodes:
                if is_non_compute_node(node) and \
                    (is_transparent_input_node(node, partition.nodes, remove_node) or
                     is_transparent_output_node(node, partition.nodes, remove_node)):
                    remove_node.add(node)

            if len(remove_node) != 0:
                partition.nodes = partition.nodes - remove_node

    def partition_and_fuse(self) -> GraphModule:
        partitions = self.propose_partitions()
        fused_gm = self.fuse_partitions(partitions)
        return fused_gm

def split_partition(partition: Partition) -> List[Partition]:
    return [partition]

def generate_partitions_from_list_of_nodes(
    graph_module: torch.fx.GraphModule,
    pattern_list: Optional[List[List[torch.fx.Node]]] = None,
    op_support: Optional[OperatorSupportBase] = None,
) -> List[Partition]:
    # assert False
    final_op_support: Optional[OperatorSupportBase] = op_support

    # pattern_list is None
    if pattern_list is not None:
        # Tag all the nodes in these patterns
        for node_list in pattern_list:
            for node in node_list:
                node.meta["match"] = True

        class MatchTag(OperatorSupportBase):
            def is_node_supported(self, submodules, node: torch.fx.Node) -> bool:
                return node.meta.get("match", False)

        final_op_support = (
            MatchTag()
            if final_op_support is None
            else any_chain(final_op_support, MatchTag())
        )

    assert (
        final_op_support is not None
    ), "Did not give a pattern or OperatorSupportBase instance to partition with"

    # import pdb; pdb.set_trace()
    # TODO: pass nn.module with same node_ids, list and partition

    # Run the CapabilityBasedPartitioner to return the largest possible
    # subgraphs containing the nodes with the tags
    # capability_partitioner = CapabilityBasedPartitioner(
    #     graph_module,
    #     final_op_support,
    #     allows_single_node_partition=True,
    # )
    # partition_list = capability_partitioner.propose_partitions()

    capability_partitioner = CapabilityPartitioner(
        graph_module,
        final_op_support,
        allows_single_node_partition=True,
    )
    partition_list = capability_partitioner.propose_sliced_partitions()

    # 1. copy find_cycle function, find who are the root nodes
    # TODO: follow the find_cycle function, traverse to a partition with max num nodes?

    # 1 proposed partition

    # import pdb; pdb.set_trace()

    # TODO: add Node.meta or other propoerties to re-partition
    # Remove the metadata field we added
    # node.meta
    # dict_keys(['stack_trace', 'nn_module_stack', 'source_fn_stack', 'original_aten', 'from_node', 'seq_nr', 'val', 'tensor_meta', 'debug_handle', 'quant_attrs'])
    for partition in partition_list:
        for node in partition.nodes:
            # print(node.meta['nn_module_stack'].keys())
            stack_str = ",".join(node.meta['nn_module_stack'].keys())
            print(f"Node: [{node.name}], nn_module_stack: {stack_str}")
            # print(f"Node: [{node.name}], stack_trace: [{node.meta['stack_trace']}], nn_module_stack: {stack_str}")
            # if f"{nn_module_stack_key_prefix}7" in node.meta['nn_module_stack'].keys():
            #     print("FOUND node in layer 7")
            # import pdb; pdb.set_trace()
            node.meta.pop("match", False)

    # # ## update partitions
    # nn_module_stack_key_prefix = "L__self___layers_"
    # partitions_by_id: Dict[int, Partition] = {}  # mapping from partition_id to partition
    # orig_partition = partition_list[0]
    # partitions_by_id[orig_partition.id] = orig_partition
    # n_layers = 2
    # for i in range(n_layers):
    #     id = orig_partition.id+1+i
    #     partitions_by_id[id] = Partition(id=id, nodes=[])

    # # for partition in partition_list:
    # #     print(f"DX Proposed partition: {partition.id}, num nodes: {len(partition.nodes)}")
    # # return partition_list

    # # # import pdb; pdb.set_trace()
    # # orig_partition = partition_list[0]
    # # new_partition = Partition(id=orig_partition.id+1)
    # # self_nodes = []

    # # # Dependency cycle: https://www.internalfb.com/code/fbsource/[7e5b3c47c5dd]/fbcode/caffe2/torch/fx/passes/utils/fuser_utils.py?lines=43
    # # # generate QNN size = 0
    # new_partition_node_id_pairs = []
    # for node in orig_partition.nodes:
    #     for i in range(n_layers):
    #         if f"{nn_module_stack_key_prefix}{i}" in node.meta['nn_module_stack'].keys():
    #             new_partition_node_id_pairs.append((orig_partition.id+1+i, node))
    #     # if "L__self___norm" in node.meta['nn_module_stack'].keys():
    #     #     new_partition_nodes.append(node)
    #     # if "L__self___output" in node.meta['nn_module_stack'].keys():
    #     #     new_partition_nodes.append(node)
    #     # if len(node.meta['nn_module_stack'].keys()) == 1:
    #     #     self_nodes.append(node)

    # # print(f"Number of L__self__ ONLY nodes: {len(self_nodes)}")

    # for partition_id, node in new_partition_node_id_pairs:
    #     partitions_by_id[partition_id].add_node(node)
    #     orig_partition.remove_node(node)
    # partition_list = list(partitions_by_id.values())

    for partition in partition_list:
        print(f"DX Proposed partition: {partition.id}, num nodes: {len(partition.nodes)}")
    return partition_list

    # return [orig_partition, new_partition]




# def generate_pattern_op_partitions(
#     graph_module: torch.fx.GraphModule,
#     patterns: Optional[List[torch.fx.Graph]] = None,
#     partitions_list: Optional[List[List[torch.fx.Node]]] = None,
#     op_support: Optional[OperatorSupportBase] = None,
#     ignore_literals: bool = False,
# ) -> List[Partition]:
#     """
#     Args:
#         graph_module: Module that we want to partition
#         patterns: A list of patterns in the form of torch.fx.Graph. These graphs
#             can be obtained through the `graph` field from a GraphModule obtained by
#             exir.capture (recommended) or symbolic tracing (which might not result
#             in an accurate edge dialect graph), or by manual crafting a graph
#             module.
#         partitions_list: A list of node lists whose nodes are intended to be tagged
#             along with the nodes detected by the pattern matcher.
#         op_support: A OperatorSupportBase that can be created in the following ways:
#             - Subclassing it directly and implementing is_node_supported()
#             - Getting the result of create_op_support()
#             - Getting the result of create_pattern_support()
#             - Multiple OperatorSupportBase classes chained together with chain()

#     Returns
#         A list of partitions (largest possible subgraphs) containing nodes are
#         supported by the given OperatorSupportBase object
#     """
#     final_op_support: Optional[OperatorSupportBase] = op_support

#     if patterns is not None:
#         # Find all patterns in the graph (even if they're invalid)
#         matches = []
#         for pattern in patterns:
#             logging.debug(f"Finding matches for pattern: {pattern}")
#             subgraph_matcher = SubgraphMatcher(pattern, ignore_literals=ignore_literals)
#             matches.extend(subgraph_matcher.match(graph_module.graph))

#         # Tag all the nodes in these patterns
#         for match in matches:
#             for node_in_pattern, node_in_graph in match.nodes_map.items():
#                 if (
#                     node_in_pattern.op != "placeholder"
#                     and node_in_graph.op != "placeholder"
#                 ):
#                     node_in_graph.meta["match"] = True

#     if partitions_list:
#         for node_list in partitions_list:
#             for node in node_list:
#                 node.meta["match"] = True

#     class MatchTag(OperatorSupportBase):
#         def is_node_supported(self, submodules, node: torch.fx.Node) -> bool:
#             return node.meta.get("match", False)

#     final_op_support = (
#         MatchTag()
#         if final_op_support is None
#         else any_chain(final_op_support, MatchTag())
#     )

#     assert (
#         final_op_support is not None
#     ), "Did not give a pattern or OperatorSupportBase instance to partition with"

#     # Run the CapabilityBasedPartitioner to return the largest possible
#     # subgraphs containing the nodes with the tags
#     capability_partitioner = CapabilityBasedPartitioner(
#         graph_module,
#         final_op_support,
#         allows_single_node_partition=True,
#     )
#     partition_list = capability_partitioner.propose_partitions()

#     # Remove the metadata field we added
#     for partition in partition_list:
#         for node in partition.nodes:
#             node.meta.pop("match", False)
#     return partition_list

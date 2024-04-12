# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import List, Optional

import torch
from torch.fx.passes.infra.partitioner import CapabilityBasedPartitioner, Partition
from torch.fx.passes.operator_support import any_chain, OperatorSupportBase
from torch.fx.passes.utils.matcher_utils import SubgraphMatcher

# def partition_nodes(
#     graph_module: torch.fx.GraphModule,
# ) -> List[Partition]:
#     return []

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
    capability_partitioner = CapabilityBasedPartitioner(
        graph_module,
        final_op_support,
        allows_single_node_partition=True,
    )
    partition_list = capability_partitioner.propose_partitions()

    # 1. copy find_cycle function, find who are the root nodes
    # TODO: follow the find_cycle function, traverse to a partition with max num nodes?

    # 1 proposed partition

    # import pdb; pdb.set_trace()

    # TODO: add Node.meta or other propoerties to re-partition
    # Remove the metadata field we added
    # node.meta
    # dict_keys(['stack_trace', 'nn_module_stack', 'source_fn_stack', 'original_aten', 'from_node', 'seq_nr', 'val', 'tensor_meta', 'debug_handle', 'quant_attrs'])
    nn_module_stack_key_prefix = "L__self___layers_"
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
    # import pdb; pdb.set_trace()
    orig_partition = partition_list[0]
    new_partition = Partition(id=orig_partition.id+1)
    new_partition_nodes = []
    self_nodes = []

    # Dependency cycle: https://www.internalfb.com/code/fbsource/[7e5b3c47c5dd]/fbcode/caffe2/torch/fx/passes/utils/fuser_utils.py?lines=43
    # generate QNN size = 0
    for node in orig_partition.nodes:
        if f"{nn_module_stack_key_prefix}7" in node.meta['nn_module_stack'].keys():
            new_partition_nodes.append(node)
        if "L__self___norm" in node.meta['nn_module_stack'].keys():
            new_partition_nodes.append(node)
        if "L__self___output" in node.meta['nn_module_stack'].keys():
            new_partition_nodes.append(node)
        if len(node.meta['nn_module_stack'].keys()) == 1:
            self_nodes.append(node)

    print(f"Number of L__self__ ONLY nodes: {len(self_nodes)}")

    for node in new_partition_nodes:
        new_partition.add_node(node)
        orig_partition.remove_node(node)

    return [orig_partition, new_partition]




def generate_pattern_op_partitions(
    graph_module: torch.fx.GraphModule,
    patterns: Optional[List[torch.fx.Graph]] = None,
    partitions_list: Optional[List[List[torch.fx.Node]]] = None,
    op_support: Optional[OperatorSupportBase] = None,
    ignore_literals: bool = False,
) -> List[Partition]:
    """
    Args:
        graph_module: Module that we want to partition
        patterns: A list of patterns in the form of torch.fx.Graph. These graphs
            can be obtained through the `graph` field from a GraphModule obtained by
            exir.capture (recommended) or symbolic tracing (which might not result
            in an accurate edge dialect graph), or by manual crafting a graph
            module.
        partitions_list: A list of node lists whose nodes are intended to be tagged
            along with the nodes detected by the pattern matcher.
        op_support: A OperatorSupportBase that can be created in the following ways:
            - Subclassing it directly and implementing is_node_supported()
            - Getting the result of create_op_support()
            - Getting the result of create_pattern_support()
            - Multiple OperatorSupportBase classes chained together with chain()

    Returns
        A list of partitions (largest possible subgraphs) containing nodes are
        supported by the given OperatorSupportBase object
    """
    final_op_support: Optional[OperatorSupportBase] = op_support

    if patterns is not None:
        # Find all patterns in the graph (even if they're invalid)
        matches = []
        for pattern in patterns:
            logging.debug(f"Finding matches for pattern: {pattern}")
            subgraph_matcher = SubgraphMatcher(pattern, ignore_literals=ignore_literals)
            matches.extend(subgraph_matcher.match(graph_module.graph))

        # Tag all the nodes in these patterns
        for match in matches:
            for node_in_pattern, node_in_graph in match.nodes_map.items():
                if (
                    node_in_pattern.op != "placeholder"
                    and node_in_graph.op != "placeholder"
                ):
                    node_in_graph.meta["match"] = True

    if partitions_list:
        for node_list in partitions_list:
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

    # Run the CapabilityBasedPartitioner to return the largest possible
    # subgraphs containing the nodes with the tags
    capability_partitioner = CapabilityBasedPartitioner(
        graph_module,
        final_op_support,
        allows_single_node_partition=True,
    )
    partition_list = capability_partitioner.propose_partitions()

    # Remove the metadata field we added
    for partition in partition_list:
        for node in partition.nodes:
            node.meta.pop("match", False)
    return partition_list

# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import copy
from collections import defaultdict
from typing import Any, Dict, List

import executorch.backends.qualcomm.python.PyQnnManagerAdaptor as PyQnnManager
import torch
from executorch.backends.qualcomm.builders import node_visitor
from executorch.backends.qualcomm.qnn_preprocess import QnnBackend
from executorch.backends.qualcomm.utils.utils import generate_qnn_executorch_option, draw_graph

from executorch.exir.backend.backend_details import CompileSpec
from executorch.exir.backend.canonical_partitioners.pattern_op_partitioner import (
    generate_partitions_from_list_of_nodes,
)
from executorch.exir.backend.partitioner import (
    DelegationSpec,
    Partitioner,
    PartitionResult,
)
from executorch.exir.backend.utils import tag_constant_data
from torch.fx.passes.infra.partitioner import Partition
from torch.fx.passes.operator_support import OperatorSupportBase

from .common_defs import allow_list_operator, not_supported_operator

READ_QNN_BLOB = True


class QnnOperatorSupport(OperatorSupportBase):
    def __init__(
        self,
        edge_program: torch.export.ExportedProgram,
        compiler_specs,
        skip_node_id_set: set = None,
        skip_node_op_set: set = None,
    ):
        self.node_visitors = node_visitor.get_node_visitors(edge_program)

        self.skip_node_op_builder_set = set()
        if skip_node_op_set is not None:
            self.skip_node_op_builder_set = set(
                [
                    self.node_visitors[val]
                    for val in skip_node_op_set
                    if val in self.node_visitors
                ]
            )

        self.skip_node_id_set = skip_node_id_set
        self.nodes_to_wrappers = self.nodes_to_wrappers = defaultdict(dict)
        self.qnn_manager = PyQnnManager.QnnManager(
            generate_qnn_executorch_option(compiler_specs)
        )

        self.qnn_manager.Init()

    def is_node_supported(self, _, node: torch.fx.Node) -> bool:
        # Enforce 1 QNN blob
        if READ_QNN_BLOB:
            return True

        if node.op != "call_function" or node.target in not_supported_operator:
            return False

        if node.target in allow_list_operator:
            return True

        if self.skip_node_id_set is not None and node.name in self.skip_node_id_set:
            print(f"[QNN Partitioner Op Support]: {node.target.__name__} | Skipped")
            return False

        if (
            self.skip_node_op_builder_set is not None
            and self.node_visitors[node.target.__name__]
            in self.skip_node_op_builder_set
        ):
            print(f"[QNN Partitioner Op Support]: {node.target.__name__} | Skipped")
            return False

        supported = False
        op_wrapper = self.node_visitors[node.target.__name__].define_node(
            node, self.nodes_to_wrappers
        )

        op_wrapper_list = []
        if isinstance(op_wrapper, List):
            op_wrapper_list.extend(op_wrapper)
        else:
            op_wrapper_list.append(op_wrapper)

        if op_wrapper is not None:
            supported = self.qnn_manager.IsNodeSupportedByBackend(
                [op_wrapper.GetOpWrapper() for op_wrapper in op_wrapper_list]
            )

        self.nodes_to_wrappers.clear()
        # Enforce 1 QNN blob
        if node.target.__name__ in ['aten.add.Tensor', 'aten.permute_copy.default', 'aten.embedding.default']:
            supported = True
        print(f"[QNN Partitioner Op Support]: {node.target.__name__} | {supported}")
        return supported


class QnnPartitioner(Partitioner):
    def __init__(
        self,
        compiler_specs: List[CompileSpec],
        skip_node_id_set: set = None,
        skip_node_op_set: set = None,
    ):
        self.compiler_specs_snapshot = copy.deepcopy(compiler_specs)

        self.delegation_spec = DelegationSpec(
            QnnBackend.__name__, self.compiler_specs_snapshot
        )
        self.partition_tags: Dict[str, DelegationSpec] = {}
        self.skip_node_id_set = skip_node_id_set
        self.skip_node_op_set = skip_node_op_set

    def generate_partitions(
        self, edge_program: torch.export.ExportedProgram
    ) -> List[Any]:
        self.op_support_checker = QnnOperatorSupport(
            edge_program,
            self.compiler_specs_snapshot,
            self.skip_node_id_set,
            self.skip_node_op_set,
        )

        title = "cria"
        path = "/tmp"
        draw_graph(title, path, edge_program.graph_module)
        print(f"Saved CPU edge program to {path}/{title}.svg {__file__}", flush=True)

        return generate_partitions_from_list_of_nodes(
            edge_program.graph_module,
            op_support=self.op_support_checker,
        )

    def tag_nodes(self, partitions: List[Partition]) -> None:
        # assert False
        default_tag = None
        for partition in partitions:
            print(f"DX tagging nodes for partition {partition.id}", flush=True)
            for node in partition.nodes:
                # print(f"DX tagging node {node.name}")
                delegation_tag = f"qnn_{partition.id}"
                if default_tag is None:
                    default_tag = delegation_tag
                # # DX
                # # Enforce using the same partition tag
                # delegation_tag = default_tag
                # node.meta["delegation_tag"] = delegation_tag
                self.partition_tags[delegation_tag] = self.delegation_spec

        print(f"DX Partition tags: {self.partition_tags}", flush=True)

    # override
    def partition(self, edge_program: torch.export.ExportedProgram) -> PartitionResult:
        partitions = self.generate_partitions(edge_program)
        if len(partitions) != 0:
            self.tag_nodes(partitions)
            tag_constant_data(edge_program)
        for node in edge_program.graph_module.graph.nodes:
            if hasattr(node, "meta"):
                # pop certain keys in meta for not affecting the passes in compilation
                # TODO: need to put property name in common definitions
                node.meta.pop("axis_order", "")
        return PartitionResult(
            tagged_exported_program=edge_program, partition_tags=self.partition_tags
        )

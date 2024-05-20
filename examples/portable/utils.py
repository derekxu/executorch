# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os

from typing import Any, Dict, Optional, Tuple, Union

import executorch.exir as exir

import torch
from executorch.exir import EdgeProgramManager, ExecutorchProgramManager, to_edge
from executorch.exir.tracer import Value
from torch._export import capture_pre_autograd_graph
from torch.export import export, ExportedProgram
from executorch.backends.qualcomm.utils.utils import draw_graph

import torch.fx
from executorch.exir import delegate
from torch.fx.node import _format_arg, _get_qualified_name


_EDGE_COMPILE_CONFIG = exir.EdgeCompileConfig(
    _check_ir_validity=True,
    _skip_dim_order=True,  # TODO(T189114319): Reuse dim order op after solving the ios oss issue
)

__all__ = ["GraphVisualizer"]
try:
    import pydot

    HAS_PYDOT = True
except ImportError:
    print("Failed to import pydot for GraphVisualizer. Please install pydot.")
    HAS_PYDOT = False


_COLOR_MAP = {
    "placeholder": '"AliceBlue"',
    "call_module": "LemonChiffon1",
    "get_param": "Yellow2",
    "get_attr": "LightGrey",
    "output": "PowderBlue",
}

_TEMPLATE = {
    "shape": "record",
    "fillcolor": "#CAFFE3",
    "style": '"filled,rounded"',
    "fontcolor": "#000000",
}

class GraphVisualizer:
    def __init__(
        self,
        graph_module: torch.fx.GraphModule,
        name: str = None,
        is_compact=False,
        expand_sub_modules=False,
    ):
        self.graph_module = graph_module
        self.name = graph_module.__class__.__name__ if name is None else name
        self.is_compact = is_compact
        self.expand_sub_modules = expand_sub_modules
        self.dot_graph = pydot.Dot(name, rankdir="TB")
        self.dot_graph.set_dpi(60)
        # build a dict of submodules to generate subgraphs
        self._graph_sub_modules = {}
        for nm in self.graph_module.named_modules():
            self._graph_sub_modules[nm[0]] = nm[1]

        self._get_subgraphs(self.dot_graph, self.graph_module, self.name)

    def get_dot_graph(self) -> pydot.Dot:
        return self.dot_graph

    def _get_subgraphs(
        self, pydot_graph: pydot.Graph, gm: torch.fx.GraphModule, module_name: str
    ) -> Any:
        node_names = []  # names are used to draw edges
        for node in gm.graph.nodes:
            style = self._get_node_style(node)
            node_name = self._get_node_name(node, module_name)
            dot_node = pydot.Node(
                node_name, label=self._get_node_label(node, module_name), **style
            )
            pydot_graph.add_node(dot_node)
            node_names.append(node_name)

            if (
                node.op == "call_function"
                and self._typename(node.target)
                == "torch.ops.higher_order.executorch_call_delegate"
                and self.expand_sub_modules
            ):
                for arg in node.args:
                    arg_name = str(arg)
                    if arg_name in self._graph_sub_modules:
                        sub_module = self._graph_sub_modules[arg_name]
                        sub_module_style = _TEMPLATE.copy()
                        sub_module_style["fillcolor"] = "MistyRose1"
                        call_module_sub_graph = pydot.Cluster(
                            node_name,
                            label=node_name,
                            rank="same",
                            **sub_module_style,
                        )
                        sub_graph, sg_node_names = self._get_subgraphs(
                            call_module_sub_graph,
                            sub_module.original_module.graph_module,
                            str(arg),
                        )
                        pydot_graph.add_subgraph(sub_graph)

                        # add an edge between call_module node and sub_graph
                        pydot_graph.add_edge(pydot.Edge(dot_node, sg_node_names[0]))
                        # this is used if we need to add an edge from output of the submodule to the call_submodule node.
                        # self.dot_graph2.add_edge(pydot.Edge(sub_graph_nodes[-1], dot_node))

        for node in gm.graph.nodes:
            for user in node.users:
                pydot_graph.add_edge(
                    pydot.Edge(
                        self._get_node_name(node, module_name),
                        self._get_node_name(user, module_name),
                    )
                )

        return pydot_graph, node_names

    def _get_node_style(self, node: torch.fx.Node) -> Dict[str, str]:
        template = _TEMPLATE.copy()
        if (
            node.op == "call_function"
            and node.target == delegate.executorch_call_delegate
        ):
            template["fillcolor"] = "MistyRose2"
        elif node.op in _COLOR_MAP:
            template["fillcolor"] = _COLOR_MAP[node.op]

        return template

    def _get_node_label(self, node: torch.fx.Node, module_name: str) -> str:
        def _get_target(target):
            ret = self._typename(target)
            ret = ret.removeprefix("executorch.exir.dialects.edge._ops.")
            return ret

        def _get_str_for_args_kwargs(arg):
            if isinstance(arg, tuple):
                prefix, suffix = r" | args=(\l", r",\n)\l"
                arg_strs_list = [_format_arg(a, max_list_len=8) for a in arg]
            elif isinstance(arg, dict):
                prefix, suffix = r" | kwargs={\l", r",\n}\l"
                arg_strs_list = [
                    f"{k}: {_format_arg(v, max_list_len=8)}" for k, v in arg.items()
                ]
            else:  # Fall back to nothing in unexpected case.
                return ""

            # Strip out node names if requested.
            if self.is_compact:
                arg_strs_list = [a for a in arg_strs_list if "%" not in a]
            if len(arg_strs_list) == 0:
                return ""
            arg_strs = prefix + r", ".join(arg_strs_list) + suffix
            return arg_strs.replace("{", r"\{").replace("}", r"\}")

        label = f"{self._get_node_name(node, module_name)} | {node.op}"
        if not self.is_compact:
            label += f" | {_get_target(node.target)}"
            if len(node.args) > 0:
                label += _get_str_for_args_kwargs(node.args)
            if len(node.kwargs) > 0:
                label += _get_str_for_args_kwargs(node.kwargs)
            val = node.meta.get("val")
            if isinstance(val, torch.Tensor):
                label += f" | {val.dtype} {list(val.shape)}"
            qparams = node.meta.get("output_qparams")
            if qparams is not None and len(qparams) == 1:
                qparam = qparams[0]
                label += (
                    f" | scale={qparam['scale']}, zero_point={qparam['zero_point']}"
                )
            if node.target == delegate.executorch_call_delegate:
                if (tag := node.meta.get("delegation_tag")) is not None:
                    label += f" | {tag}"
        return "{" + label + "}"

    def _get_node_name(self, node: torch.fx.Node, module_name: str):
        name = f"%{node.name} ({module_name})"
        return name

    def _typename(self, target: Any) -> str:
        if isinstance(target, torch.nn.Module):
            ret = torch.typename(target)
        elif isinstance(target, str):
            ret = target
        else:
            ret = _get_qualified_name(target)

        # Escape "{" and "}" to prevent dot files like:
        # https://gist.github.com/SungMinCho/1a017aab662c75d805c5954d62c5aabc
        # which triggers `Error: bad label format (...)` from dot
        return ret.replace("{", r"\{").replace("}", r"\}")



def _to_core_aten(
    model: Union[torch.fx.GraphModule, torch.nn.Module],
    example_inputs: Tuple[Value, ...],
    dynamic_shapes: Optional[Union[Dict[str, Any], Tuple[Any]]] = None,
    verbose=True,
) -> ExportedProgram:
    # post autograd export. eventually this will become .to_core_aten
    if not isinstance(model, torch.fx.GraphModule) and not isinstance(
        model, torch.nn.Module
    ):
        raise ValueError(
            f"Expected passed in model to be an instance of fx.GraphModule, got {type(model)}"
        )
    core_aten_ep = export(model, example_inputs, dynamic_shapes=dynamic_shapes)
    if verbose:
        logging.info(f"Core ATen graph:\n{core_aten_ep.graph}")
    return core_aten_ep


def _core_aten_to_edge(
    core_aten_exir_ep: ExportedProgram,
    edge_constant_methods: Optional[Dict[str, Any]] = None,
    edge_compile_config=None,
    verbose=True,
) -> EdgeProgramManager:
    if not edge_compile_config:
        edge_compile_config = exir.EdgeCompileConfig(
            _check_ir_validity=False,  # quant ops currently break ir verification
            _skip_dim_order=True,  # TODO(T182928844): dim order ops can not delegate to backend
        )
    edge_manager: EdgeProgramManager = to_edge(
        core_aten_exir_ep,
        constant_methods=edge_constant_methods,
        compile_config=edge_compile_config,
    )
    if verbose:
        logging.info(f"Exported graph:\n{edge_manager.exported_program().graph}")
    return edge_manager


def export_to_edge(
    model: Union[torch.fx.GraphModule, torch.nn.Module],
    example_inputs: Tuple[Value, ...],
    dynamic_shapes: Optional[Union[Dict[str, Any], Tuple[Any]]] = None,
    edge_constant_methods: Optional[Dict[str, Any]] = None,
    edge_compile_config=_EDGE_COMPILE_CONFIG,
    verbose=True,
) -> EdgeProgramManager:
    core_aten_ep = _to_core_aten(model, example_inputs, dynamic_shapes, verbose=verbose)
    return _core_aten_to_edge(
        core_aten_ep, edge_constant_methods, edge_compile_config, verbose=verbose
    )


def export_to_exec_prog(
    model: Union[torch.fx.GraphModule, torch.nn.Module],
    example_inputs: Tuple[Value, ...],
    dynamic_shapes: Optional[Union[Dict[str, Any], Tuple[Any]]] = None,
    edge_constant_methods: Optional[Dict[str, Any]] = None,
    edge_compile_config=_EDGE_COMPILE_CONFIG,
    backend_config=None,
) -> ExecutorchProgramManager:
    m = model.eval()
    # pre-autograd export. eventually this will become torch.export
    m = capture_pre_autograd_graph(m, example_inputs)

    core_aten_ep = _to_core_aten(m, example_inputs, dynamic_shapes)

    edge_m = _core_aten_to_edge(
        core_aten_ep, edge_constant_methods, edge_compile_config
    )

    exec_prog = edge_m.to_executorch(backend_config)
    return exec_prog


def save_pte_program(
    prog: ExecutorchProgramManager, model_name: str, output_dir: str = ""
) -> None:
    if model_name.endswith(".pte"):
        filename = model_name
    else:
        filename = os.path.join(output_dir, f"{model_name}.pte")

    draw_graph("llm_graph", "/tmp", prog.exported_program().graph_module)

    try:
        with open(filename, "wb") as file:
            prog.write_to_file(file)
            logging.info(f"Saved exported program to {filename}")
    except Exception as e:
        logging.error(f"Error while saving to {filename}: {e}")

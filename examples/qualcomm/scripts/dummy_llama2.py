# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
import os
import re
from multiprocessing.connection import Client

import numpy as np
import torch
from executorch.examples.models.llama2 import Llama2Model, INPUTS, MODEL_NAME
from executorch.examples.qualcomm.scripts.utils import (
    build_executorch_binary,
    make_output_dir,
    setup_common_args_and_variables,
    SimpleADB,
)

INPUT_LEN = len(INPUTS)


def create_device_inputs(example_inputs, use_kv_cache, filename):
    print(f"Length of example inputs: {len(example_inputs)}")
    for i,inp in enumerate(example_inputs):
        print(f"Input {i}: {inp.size()}")
    inputs = [inp.to(torch.int32) for inp in example_inputs]
    input_list = ""
    if use_kv_cache:
        for i, d in enumerate(inputs[0]):
            if type(d) == list:
                d = torch.stack(d)
            np_inp = d.numpy()
            np_inp.tofile(f"{args.artifact}/{filename}")
            print(f"input: {np_inp}")
            input_list = f"input_0_{i}.raw "
    else:
        np_inp = inputs[0].numpy()
        np_inp.tofile(f"{args.artifact}/{filename}")
        print(f"input: {np_inp}")
        input_list = f"{filename}"
    input_list += "\n"
    return tuple(inputs), input_list


if __name__ == "__main__":
    parser = setup_common_args_and_variables()
    parser.add_argument(
        "-a",
        "--artifact",
        help="path for storing generated artifacts by this example. Default ./dummy_llama2",
        default="./dummy_llama2",
        type=str,
    )

    # TODO kv cache is not yet enabled
    parser.add_argument(
        "-kv",
        "--use_kv_cache",
        default=False,
        action="store_true",
        help="Whether or not to export a model using kv cache",
    )

    # NOTE: use -P as default
    parser.add_argument(
        "-P",
        "--ptq",
        help="If specified, will do PTQ.",
        action="store_true",
        default=False,
    )

    # QNN_SDK_ROOT might also be an argument, but it is used in various places.
    # So maybe it's fine to just use the environment.
    if "QNN_SDK_ROOT" not in os.environ:
        raise RuntimeError("Environment variable QNN_SDK_ROOT must be set")
    print(f"QNN_SDK_ROOT={os.getenv('QNN_SDK_ROOT')}")

    if "LD_LIBRARY_PATH" not in os.environ:
        print(
            "[Warning] LD_LIBRARY_PATH is not set. If errors like libQnnHtp.so "
            "not found happen, please follow setup.md to set environment."
        )
    else:
        print(f"LD_LIBRARY_PATH={os.getenv('LD_LIBRARY_PATH')}")

    args = parser.parse_args()

    # ensure the working directory exist.
    os.makedirs(args.artifact, exist_ok=True)

    instance = Llama2Model(use_kv_cache=args.use_kv_cache)

    use_fp16 = False if args.ptq else True

    quant_mode = "fp16"
    if not use_fp16:
        quant_mode = "int8"

    filename = f"input_len_{INPUT_LEN}_{quant_mode}_0_0.raw"

    inputs, input_list = create_device_inputs(
        instance.get_example_inputs(), args.use_kv_cache, filename
    )

    pte_filename = f"{MODEL_NAME}_len_{INPUT_LEN}_{quant_mode}"

    build_executorch_binary(
        instance.get_eager_model().eval(),
        inputs,
        args.model,
        f"{args.artifact}/{pte_filename}",
        inputs,
        custom_annotations=(),
        use_fp16=use_fp16,
    )
    for i,inp in enumerate(instance.get_example_inputs()):
        print(f"Input {i}: {inp.size()}")
    print(f"PTE: {args.artifact}/{pte_filename}.pte")
    input_list_file = f"{args.artifact}/input_list_len_{INPUT_LEN}_{quant_mode}.txt"
    with open(input_list_file, "w") as f:
        f.write(input_list)
        f.flush()
    print(f"input_list: {input_list}")
    print(f"input_list file: {input_list_file}")
    # adb = SimpleADB(
    #     qnn_sdk=os.getenv("QNN_SDK_ROOT"),
    #     artifact_path=f"{args.build_folder}",
    #     pte_path=f"{args.artifact}/{pte_filename}.pte",
    #     workspace=f"/data/local/tmp/executorch/{pte_filename}",
    #     device_id=args.device,
    #     host_id=args.host,
    #     soc_model=args.model,
    # )
    # adb.push(inputs=inputs, input_list=input_list)

    # adb.execute()

    # # collect output data
    # output_data_folder = f"{args.artifact}/outputs"
    # make_output_dir(output_data_folder)

    # output_raws = []

    # def post_process():
    #     for f in sorted(
    #         os.listdir(output_data_folder), key=lambda f: int(f.split("_")[1])
    #     ):
    #         filename = os.path.join(output_data_folder, f)
    #         if re.match(r"^output_[0-9]+_[1-9].raw$", f):
    #             os.remove(filename)
    #         else:
    #             output = np.fromfile(filename, dtype=np.float32)
    #             output_raws.append(output)

    # adb.pull(output_path=args.artifact, callback=post_process)

    # x86_golden = instance.get_eager_model().eval()(inputs[0])
    # device_output = torch.from_numpy(output_raws[0]).reshape(x86_golden.size())
    # result = torch.all(torch.isclose(x86_golden, device_output, atol=1e-2)).tolist()

    # if args.ip and args.port != -1:
    #     with Client((args.ip, args.port)) as conn:
    #         conn.send(
    #             json.dumps(
    #                 {
    #                     "is_close": result,
    #                 }
    #             )
    #         )
    # else:
    #     print(f"is_close? {result}")
    #     print(f"x86_golden {x86_golden}")
    #     print(f"device_out {device_output}")

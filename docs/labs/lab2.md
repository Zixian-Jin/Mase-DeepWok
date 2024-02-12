# Lab2

1. Explain the functionality of `report_graph_analysis_pass` and its printed jargons such as `placeholder`, `get_attr` … You might find the doc of `torch.fx` useful. 

    1.1 Functionality of `report_graph_analysis_pass`

    It takes a `MaseGraph` type as input, analyses the graph and print an overview of the model in a table.
    ```Python
    for node in graph.fx_graph.nodes:
        if node.meta["mase"].module is not None:
            layer_types.append(node.meta["mase"].module)
    ```
    It traverses all nodes in the graph, retrieves their metadata. Every node is of the type `torch.fx.node`. The report consists of three sections:
    *	Architecture of the graph, which is simply obtained by printing `graph.fx_graph`
    *	“Network Overview”, which counts the appearance of each fx operation types (`placeholder`, `get_attr`, etc.)
    *	“Layer Types”, which displays the module type for each node (BatchNorm, ReLU, Linear, etc.)
    
    1.2	Terminologies
    These jargons are the operation types of each node, which is one-to-one mapped to the code in the `forward` definition of the model.. As the nodes here are of the type `torch.fx.Node`, the definition of the jargons are listed in the [PyTorch documentation](https://pytorch.org/docs/stable/fx.html#torch.fx.Node):
    * `placeholder`: a function input
    * `get_attr`: operations that retrieve a parameter from the module hierarchy.
    * `call_function`: operations that apply a free function to some values. e.g., `x = 2*x`
    * `call_module`: operations that apply a module in the module hierarchy’s forward() method, e.g., `x = torch.nn.Linear(8,5)(x)`
    * `call_method`: operations that call a method on a value
    * `output`: operations that simply produce model output.


2.	What are the functionalities of `profile_statistics_analysis_pass` and `report_node_meta_param_analysis_pass` respectively?
    * The pass `profile_statistics_analysis_pass` collects statistics of parameters and activations, and save them to node's metadata.
    ```
    Profiling weight statistics: 100%|██████████| 6/6 [00:00<00:00, 1496.81it/s]
    Profiling act statistics: 100%|██████████| 4/4 [00:00<00:00, 235.33it/s]
    ```
    * The pass `report_node_meta_param_analysis_pass` performs meta parameter analysis on the nodes in the graph and generate a report. It traverses the graph, and print the selected metadata for each node. The metadata is of `MaseMetadata` class and has three components: `common`, `hardware`, and `software`.

    ```Python
    # machop/chop/passes/graph/analysis/report/report_node.py
    def report_node_meta_param_analysis_pass(graph, pass_args: dict = None):
        ...
        if "common" in which_param or "all" in which_param:
            headers.append("Common Param")
        if "hardware" in which_param or "all" in which_param:
            headers.append("Hardware Param")
        if "software" in which_param or "all" in which_param:
            headers.append("Software Param")
        ...
    ```
    

    Output:
    ```
    Inspecting graph [add_common_meta_param_analysis_pass]
    INFO
    +--------------+--------------+---------------------+--------------+-----------------------------------------------------------------------------------------+
    | Node name    | Fx Node op   | Mase type           | Mase op      | Software Param                                                                          |
    +==============+==============+=====================+==============+=========================================================================================+
    | x            | placeholder  | placeholder         | placeholder  | {'results': {'data_out_0': {'stat': {}}}}                                               |
    +--------------+--------------+---------------------+--------------+-----------------------------------------------------------------------------------------+
    | seq_blocks_0 | call_module  | module              | batch_norm1d | {'args': {'bias': {'stat': {}},                                                         |
    |              |              |                     |              |           'data_in_0': {'stat': {}},                                                    |
    |              |              |                     |              |           'running_mean': {'stat': {}},                                                 |
    |              |              |                     |              |           'running_var': {'stat': {}},                                                  |
    |              |              |                     |              |           'weight': {'stat': {}}},                                                      |
    |              |              |                     |              |  'results': {'data_out_0': {'stat': {}}}}                                               |
    +--------------+--------------+---------------------+--------------+-----------------------------------------------------------------------------------------+
    | seq_blocks_1 | call_module  | module_related_func | relu         | {'args': {'data_in_0': {'stat': {'range_quantile': {'count': 512,                       |
    |              |              |                     |              |                                                     'max': 2.4510324001312256,          |
    |              |              |                     |              |                                                     'min': -1.5772733688354492,         |
    |              |              |                     |              |                                                     'range': 4.028306007385254}}}},     |
    |              |              |                     |              |  'results': {'data_out_0': {'stat': {}}}}                                               |
    +--------------+--------------+---------------------+--------------+-----------------------------------------------------------------------------------------+
    | seq_blocks_2 | call_module  | module_related_func | linear       | {'args': {'bias': {'stat': {'variance_precise': {'count': 5,                            |
    |              |              |                     |              |                                                  'mean': 0.02854445017874241,           |
    |              |              |                     |              |                                                  'variance': 0.09843380004167557}}},    |
    |              |              |                     |              |           'data_in_0': {'stat': {}},                                                    |
    |              |              |                     |              |           'weight': {'stat': {'variance_precise': {'count': 80,                         |
    |              |              |                     |              |                                                    'mean': 0.012429135851562023,        |
    |              |              |                     |              |                                                    'variance': 0.03982768952846527}}}}, |
    |              |              |                     |              |  'results': {'data_out_0': {'stat': {}}}}                                               |
    +--------------+--------------+---------------------+--------------+-----------------------------------------------------------------------------------------+
    | seq_blocks_3 | call_module  | module_related_func | relu         | {'args': {'data_in_0': {'stat': {'range_quantile': {'count': 160,                       |
    |              |              |                     |              |                                                     'max': 2.5585620403289795,          |
    |              |              |                     |              |                                                     'min': -1.5876179933547974,         |
    |              |              |                     |              |                                                     'range': 4.146180152893066}}}},     |
    |              |              |                     |              |  'results': {'data_out_0': {'stat': {}}}}                                               |
    +--------------+--------------+---------------------+--------------+-----------------------------------------------------------------------------------------+
    | output       | output       | output              | output       | {'args': {'data_in_0': {'stat': {}}}}                                                   |
    +--------------+--------------+---------------------+--------------+-----------------------------------------------------------------------------------------+
    ```
 
3.	Explain why only 1 OP is changed after the `quantize_transform_pass`?
    The additional argument `pass_args` that is passed to `quantize_transform_pass` only contains the linear layer. Thus, the pass only quantizes linear layers, which, however, only appear once in the given `jsc_tiny` model. To support this explanation, if we append `ReLU` option to `pass_args`, both the linear and relu layers in the model will be quantized:

    ```
    Quantized graph histogram:
    | Original type   | OP           |   Total |   Changed |   Unchanged |
    |-----------------+--------------+---------+-----------+-------------|
    | BatchNorm1d     | batch_norm1d |       1 |         0 |           1 |
    | Linear          | linear       |       1 |         1 |           0 |
    | ReLU            | relu         |       2 |         2 |           0 |
    | output          | output       |       1 |         0 |           1 |
    | x               | placeholder  |       1 |         0 |           1 |
    ```

4.	Write some code to traverse both `mg` and `ori_mg`. Check and comment on the nodes in these two graphs. You might find the source code for the implementation of `summarize_quantization_analysis_pass` useful.

    ```Python
    from chop.passes.graph.utils import get_mase_op, get_mase_type, get_node_actual_target
    import pandas as pd
    from tabulate import tabulate


    def get_data_cfg(node, kw):
    try:
        result = node.meta['mase'].parameters['common']['args'][kw]
    except:
        result = {'type':'NA', 'precision':'NA', 'shape':'NA'}
    return result

    headers = ['Node', 'MASE OP', 'Ori Data Type & Prec', 'Quan Data Type & Prec', 'Ori Module', 'Quan Module']
    rows = []

    for ori_n, n in zip(ori_mg.fx_graph.nodes, mg.fx_graph.nodes):
        rows.append(
            [
                ori_n.name,
                get_mase_op(ori_n),
                get_data_cfg(ori_n, 'data_in_0')['type'] + ': ' + get_data_cfg(ori_n, 'data_in_0')['precision'].__str__(),
                get_data_cfg(n, 'data_in_0')['type'] + ': ' + get_data_cfg(n, 'data_in_0')['precision'].__str__(),
                get_node_actual_target(ori_n).__class__.__name__,
                get_node_actual_target(n).__class__.__name__
            ]
        )
    print("Compare nodes:")
    print("\n" + tabulate(rows, headers=headers, tablefmt="orgtbl"))
    ```

    Output:
    ```
    Compare nodes:
    | Node         | MASE OP      | Ori Data Type & Prec   | Quan Data Type & Prec   | Ori Module   | Quan Module   |
    |--------------+--------------+------------------------+-------------------------+--------------+---------------|
    | x            | placeholder  | NA: NA                 | NA: NA                  | str          | str           |
    | seq_blocks_0 | batch_norm1d | float: [32]            | float: [32]             | BatchNorm1d  | BatchNorm1d   |
    | seq_blocks_1 | relu         | float: [32]            | integer: [8, 4]         | ReLU         | ReLUInteger   |
    | seq_blocks_2 | linear       | float: [32]            | integer: [8, 4]         | Linear       | LinearInteger |
    | seq_blocks_3 | relu         | float: [32]            | integer: [8, 4]         | ReLU         | ReLUInteger   |
    | output       | output       | NA: NA                 | NA: NA                  | str          | str           |
    ```

5.	Perform the same quantisation flow to the bigger JSC network that you have trained in lab1. You must be aware that now the `pass_args` for your custom network might be different if you have used more than the Linear layer in your network. 

    Set `model_name=jsc-zixian`, reload the model and perform quantisation pass:
    ```
    Quantized graph histogram:
    | Original type   | OP           |   Total |   Changed |   Unchanged |
    |-----------------+--------------+---------+-----------+-------------|
    | BatchNorm1d     | batch_norm1d |       6 |         0 |           6 |
    | Linear          | linear       |       5 |         5 |           0 |
    | ReLU            | relu         |       6 |         6 |           0 |
    | output          | output       |       1 |         0 |           1 |
    | x               | placeholder  |       1 |         0 |           1 |
    ```

6.	Write code to show and verify that the weights of these layers are indeed quantised. You might need to go through the source code of the implementation of the quantisation pass and also the implementation of the `Quantized Layers`.
    ```Python
    # Get input data
    test_x = iter(data_module.val_dataloader())
    xs, ys = next(test_x)

    # Find quantized relu module
    for ori_n, n in zip(ori_mg.fx_graph.nodes, mg.fx_graph.nodes):
        if (get_mase_op(n) == 'relu'):
            ori_relu = get_node_actual_target(ori_n)
            relu = get_node_actual_target(n)
        if (get_mase_op(n) == 'linear'):
            ori_linear = get_node_actual_target(ori_n)
            linear = get_node_actual_target(n)
    print('Ori_ReLU:', ori_relu(xs))
    print('ReLU:', relu(xs))
    print('Ori Linear:', ori_linear(xs))
    print('Linear:', linear(xs))
    ```
    Output:
    ```
    Ori_ReLU: 
        tensor([[0.0000, 0.8195, 2.0237, 2.1123, 1.9694, 2.1794, 0.0000, 0.2523, 0.0000,
            0.7871, 0.8023, 1.2439, 0.6504, 0.9804, 1.8541, 1.1823],
            [0.0000, 0.6081, 0.1573, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
            0.0000, 0.2638, 0.0000, 0.0000, 0.0000, 0.0334, 0.1212],
            ...])
    ReLU: 
        tensor([[0.0000, 0.8125, 2.0000, 2.1250, 2.0000, 2.1875, 0.0000, 0.2500, 0.0000,
        0.8125, 0.8125, 1.2500, 0.6250, 1.0000, 1.8750, 1.1875],
        [0.0000, 0.6250, 0.1875, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
        0.0000, 0.2500, 0.0000, 0.0000, 0.0000, 0.0625, 0.1250],
        ...])
    Ori Linear: 
        tensor([[ 1.3917, -2.0909, -1.4046, -0.6611,  2.9457],
                [ 0.0833,  0.1057,  0.6065, -0.3617, -0.0801],
                ...], grad_fn=<AddmmBackward0>)
    Linear: 
        tensor([[ 1.5156, -2.1836, -1.2461, -0.7070,  3.0664],
                [ 0.1094,  0.0586,  0.6172, -0.3516, -0.0820],
                ...], grad_fn=<AddmmBackward0>)
    ```


7.	Load your own pre-trained JSC network, and perform the quantisation using the command line interface.
 
    ```bash

    (py_adls) E:\OneDrive - Imperial College London\FYP\Mase-DeepWok\machop>python ./ch transform --config configs/examples/jsc_zixian_by_type.toml --task cls --cpu 0

    INFO     Initialising model 'jsc-zixian'...
    INFO     Initialising dataset 'jsc'...
    INFO     Project will be created at E:\OneDrive - Imperial College London\FYP\Mase-DeepWok\mase_output\jsc-tiny
    INFO     Transforming model 'jsc-zixian'...
    INFO     Loaded pytorch lightning checkpoint from ../mase_output/jsc-zixian_classification_jsc_2024-01-30/software/training_ckpts/best.ckpt
    INFO     Quantized graph histogram:
    INFO
    | Original type   | OP           |   Total |   Changed |   Unchanged |
    |-----------------+--------------+---------+-----------+-------------|
    | BatchNorm1d     | batch_norm1d |       6 |         0 |           6 |
    | Linear          | linear       |       5 |         5 |           0 |
    | ReLU            | relu         |       6 |         6 |           0 |
    | output          | output       |       1 |         0 |           1 |
    | x               | placeholder  |       1 |         0 |           1 |
    INFO     Saved mase graph to E:\OneDrive - Imperial College London\FYP\Mase-DeepWok\mase_output\jsc-tiny\software\transform\transformed_ckpt
    INFO     Transformation is completed
    ```

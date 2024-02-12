# Lab3

If following the instructions in `lab3.md`, the search result would be similar to [FIG], where the inference accuracy does not strictly decrease for lower precision configuration. 
```
# jsc_zixian, num_batches=5, both linear and relu, pretrain=False
Data_in Precision=(16, 8), Weight Precision=(16, 8), Acc=0.129524
Data_in Precision=(16, 8), Weight Precision=(8, 6), Acc=0.207143
Data_in Precision=(16, 8), Weight Precision=(8, 4), Acc=0.389524
Data_in Precision=(16, 8), Weight Precision=(4, 2), Acc=0.171429
Data_in Precision=(8, 6), Weight Precision=(16, 8), Acc=0.147619
Data_in Precision=(8, 6), Weight Precision=(8, 6), Acc=0.144762
Data_in Precision=(8, 6), Weight Precision=(8, 4), Acc=0.219048
Data_in Precision=(8, 6), Weight Precision=(4, 2), Acc=0.133333
...
```
There are basically three reasons:
- The model is loaded without pretrained checkpoints. Parameters in the model are therefore randomly initialized (the low validation accuracy also proves this). Quantization makes no sense here as quantizing parameters to lower precision may even push the weights closer to their optimal values, hence yielding even better accuracy.
- The number of batches is set small. For each precision config, the input data batch is different, which causes the validation accuracy sensitive to input data. 
- Considering the two points, to better evaluate the effect of quantization, I loaded a pretrained checkpoint, and increase the number of batches up to 50. Then the accuracy clearly decreases when using lower precision for quantization:

```
# jsc_zixian, num_batches=50, both linear and relu, pretrain=True
Data_in Precision=(16, 8), Weight Precision=(16, 8), Acc=0.415964
Data_in Precision=(16, 8), Weight Precision=(8, 6), Acc=0.422533
Data_in Precision=(16, 8), Weight Precision=(8, 4), Acc=0.423464
Data_in Precision=(16, 8), Weight Precision=(4, 2), Acc=0.292369
Data_in Precision=(8, 6), Weight Precision=(16, 8), Acc=0.408399
Data_in Precision=(8, 6), Weight Precision=(8, 6), Acc=0.417247
Data_in Precision=(8, 6), Weight Precision=(8, 4), Acc=0.405163
Data_in Precision=(8, 6), Weight Precision=(4, 2), Acc=0.307680
...
```

1.	Explore additional metrics that can serve as quality metrics for the search process. For example, you can consider metrics such as latency, model size, or the number of FLOPs (floating-point operations) involved in the model.
2.	Implement some of these additional metrics and attempt to combine them with the accuracy or loss quality metric. It's important to note that in this particular case, accuracy and loss actually serve as the same quality metric (do you know why?).
    Two metrics are implemented: latency (inference time) and model size.
    * Inference time:

    ```Python
    from time import time
    recorded_time = []  # inference latency
    for i, config in enumerate(search_spaces):
        mg, _ = quantize_transform_pass(mg, config)
        # this is the inner loop, where we also call it as a runner.
        ...
        t0 = time()
        for inputs in data_module.train_dataloader():
            ...
        t1 = time()
    time_avg = (t1 - t0)/ len(inputs)
    recorded_time.append(time_avg)
    ```

    * Model size:
    ```Python
    def getModelSize(model):
        param_size = 0
        param_sum = 0
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
            param_sum += param.nelement()
        param_size_kb = param_size / 1024
        return param_size_kb
    ```

    The metrics after each quantization are shown below:
    ```
    Data_in Precision=(16, 8), Weight Precision=(16, 8), Acc=0.129524, Latency=0.061050, Model Size=5 kB
    Data_in Precision=(16, 8), Weight Precision=(8, 6), Acc=0.207143, Latency=0.045514, Model Size=5 kB
    Data_in Precision=(16, 8), Weight Precision=(8, 4), Acc=0.389524, Latency=0.048004, Model Size=5 kB
    Data_in Precision=(16, 8), Weight Precision=(4, 2), Acc=0.171429, Latency=0.051514, Model Size=5 kB
    Data_in Precision=(8, 6), Weight Precision=(16, 8), Acc=0.147619, Latency=0.041927, Model Size=5 kB
    Data_in Precision=(8, 6), Weight Precision=(8, 6), Acc=0.144762, Latency=0.042500, Model Size=5 kB
    Data_in Precision=(8, 6), Weight Precision=(8, 4), Acc=0.219048, Latency=0.054519, Model Size=5 kB
    Data_in Precision=(8, 6), Weight Precision=(4, 2), Acc=0.133333, Latency=0.047001, Model Size=5 kB
    Data_in Precision=(8, 4), Weight Precision=(16, 8), Acc=0.188095, Latency=0.041500, Model Size=5 kB
    Data_in Precision=(8, 4), Weight Precision=(8, 6), Acc=0.102381, Latency=0.056012, Model Size=5 kB
    Data_in Precision=(8, 4), Weight Precision=(8, 4), Acc=0.109524, Latency=0.042978, Model Size=5 kB
    Data_in Precision=(8, 4), Weight Precision=(4, 2), Acc=0.198571, Latency=0.041997, Model Size=5 kB
    Data_in Precision=(4, 2), Weight Precision=(16, 8), Acc=0.159524, Latency=0.051620, Model Size=5 kB
    Data_in Precision=(4, 2), Weight Precision=(8, 6), Acc=0.180952, Latency=0.040914, Model Size=5 kB
    Data_in Precision=(4, 2), Weight Precision=(8, 4), Acc=0.074762, Latency=0.048963, Model Size=5 kB
    Data_in Precision=(4, 2), Weight Precision=(4, 2), Acc=0.164286, Latency=0.052660, Model Size=5 kB
    ```  
    It is observed that the latency is uncorrelated with param precision, and the model size remains constant. This is because the quantizer does not change the data type that the parameters are stored in software:
    ```Python
    # machop/chop/passes/graph/transforms/quantize/quantizers/integer.py
    def _integer_quantize(
        x: Tensor | ndarray, width: int, frac_width: int = None, is_signed: bool = True
    ):
        if frac_width is None:
            frac_width = width // 2

        if is_signed:
            int_min = -(2 ** (width - 1))
            int_max = 2 ** (width - 1) - 1
        else:
            int_min = 0
            int_max = 2**width - 1
        # thresh = 2 ** (width - 1)
        scale = 2**frac_width

        if isinstance(x, (Tensor, ndarray)):
            return my_clamp(my_round(x.mul(scale)), int_min, int_max).div(scale)
        elif isinstance(x, int):
            return x
        else:
            return my_clamp(my_round(x * scale), int_min, int_max) / scale
    ```
    For example, if an array x is to be quantized from FP16 to Fixed-Point<8, 4>, the integer quantizer only clamps all elements in x into a range (min, max), which is the range that Fixed-Point<8, 4> can represent. After quantization, the model parameters still have the data type FP16, and model validation still sees the same amount of FP operations on CPU or GPU. Therefore, theoretically the latency and model size should remain constant regardless of quantization configs. These metrics make more sense when the model is implemented on hardware, where parameters are indeed stored and computed as fixed-point-precision digits.


3.	Implement the brute-force search as an additional search method within the system, this would be a new search strategy in MASE.

    When executing the search command `./ch search –config …`, the script invokes `ChopCLI._run_search` method, which in turn finds the Optuna search strategy defined in `machop/chop/actions/search/strategies/optuna.py`:
    
    ```Python
    class SearchStrategyOptuna(SearchStrategyBase):
        ...
        def sampler_map(self, name):
            match name.lower():
                case "random":
                    sampler = optuna.samplers.RandomSampler()
                case "tpe":
                    sampler = optuna.samplers.TPESampler()
                ...
    ```
    The class `SearchStrategyOptuna` implements several sampler methods imported from `optuna` library including TPE sampling strategy. To implement brute-force search, just call the method `optuna.samplers.BruteForceSampler()`.

4.	Compare the brute-force search with the TPE based search, in terms of sample efficiency. Comment on the performance difference between the two search methods.

    * BruteForce-20:
    ```bash
    INFO     Building search space...
    INFO     Search started...
    F:\AdvDeepLearningSys\Mase-DeepWok\machop\chop\actions\search\strategies\optuna.py:47: ExperimentalWarning: BruteForceSampler is experimental (supported from v3.1.0). The interface can change in the future.
    sampler = optuna.samplers.BruteForceSampler()
    90%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▍              | 18/20 [04:19<00:28, 14.43s/it, 259.77/20000 seconds]
    INFO     Best trial(s):
    Best trial(s):
    |    |   number | software_metrics                   | hardware_metrics                                  | scaled_metrics                               |
    |----+----------+------------------------------------+---------------------------------------------------+----------------------------------------------|
    |  0 |       11 | {'loss': 1.267, 'accuracy': 0.533} | {'average_bitwidth': 4.0, 'memory_density': 8.0}  | {'accuracy': 0.533, 'average_bitwidth': 0.8} |
    |  1 |       13 | {'loss': 1.286, 'accuracy': 0.521} | {'average_bitwidth': 2.0, 'memory_density': 16.0} | {'accuracy': 0.521, 'average_bitwidth': 0.4} |
    INFO     Searching is completed
    ```

    * TPE-20:
    ```bash
    INFO     Building search space...
    INFO     Search started...
    100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 20/20 [04:35<00:00, 13.75s/it, 275.09/20000 seconds]
    INFO     Best trial(s):
    Best trial(s):
    |    |   number | software_metrics                   | hardware_metrics                                  | scaled_metrics                               |
    |----+----------+------------------------------------+---------------------------------------------------+----------------------------------------------|
    |  0 |       12 | {'loss': 1.308, 'accuracy': 0.509} | {'average_bitwidth': 4.0, 'memory_density': 8.0}  | {'accuracy': 0.509, 'average_bitwidth': 0.8} |
    |  1 |       13 | {'loss': 1.315, 'accuracy': 0.502} | {'average_bitwidth': 2.0, 'memory_density': 16.0} | {'accuracy': 0.502, 'average_bitwidth': 0.4} |
    INFO     Searching is completed
    ```
    * TPE-10:
    ```bash
    INFO     Building search space...
    INFO     Search started...
    100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 10/10 [02:13<00:00, 13.35s/it, 133.48/20000 seconds]
    INFO     Best trial(s):
    Best trial(s):
    |    |   number | software_metrics                   | hardware_metrics                                  | scaled_metrics                               |
    |----+----------+------------------------------------+---------------------------------------------------+----------------------------------------------|
    |  0 |        0 | {'loss': 1.024, 'accuracy': 0.649} | {'average_bitwidth': 8.0, 'memory_density': 4.0}  | {'accuracy': 0.649, 'average_bitwidth': 1.6} |
    |  1 |        4 | {'loss': 1.21, 'accuracy': 0.585}  | {'average_bitwidth': 4.0, 'memory_density': 8.0}  | {'accuracy': 0.585, 'average_bitwidth': 0.8} |
    |  2 |        7 | {'loss': 1.185, 'accuracy': 0.58}  | {'average_bitwidth': 2.0, 'memory_density': 16.0} | {'accuracy': 0.58, 'average_bitwidth': 0.4}  |
    INFO     Searching is completed                                               
    ```
    * Brute-force traverses all nodes (precision configs) in the search space. It is guaranteed to find the optimal solution `average_bitwidth: 0.8`. 
    * TPE traverses nodes in a certain algorithm. It can find the optimal solution in less trials, which is though not guaranteed.
    * When setting `n_trials=20` which exceeds the search space capacity, both brute-force and TPE finds the same optimal solution, as both traverses all nodes. When setting `n_trials=10`, TPE obtains the optimal solution 
    * For large search space, TPE is preferred since brute-force takes much time. However, for small search space, brute-force is as efficient as TPE while has guaranteed performance.

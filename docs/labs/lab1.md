# Lab1


```Python
# machop/chop/models/physical/jet_substructure/__init__.py
class JSC_Zixian(nn.Module):
    def __init__(self, info):
        super(JSC_Zixian, self).__init__()
        self.seq_blocks = nn.Sequential(
            # 1st LogicNets Layer
            nn.BatchNorm1d(16),  # input_quant       # 0
            nn.ReLU(16),  # 1
            nn.Linear(16, 32),  # linear              # 2
            nn.BatchNorm1d(32),  # output_quant       # 3
            nn.ReLU(32),  # 4

            # 2nd LogicNets Layer
            nn.Linear(32, 16),  # 5
            nn.BatchNorm1d(16),  # 6
            nn.ReLU(16),  # 7

            # 3rd LogicNets Layer
            nn.Linear(16, 8),  # 8
            nn.BatchNorm1d(8),  # 9
            nn.ReLU(8),  # 10

            # 4th LogicNets Layer
            nn.Linear(8, 8),  # 11
            nn.BatchNorm1d(8),  # 12
            nn.ReLU(8),  # 13

            # 5th LogicNets Layer
            nn.Linear(8, 5),  # 14
            nn.BatchNorm1d(5),  # 15
            nn.ReLU(5),
        )

    def forward(self, x):
        return self.seq_blocks(x)
```

```Python
# machop/chop/models/physical/__init__.py
PHYSICAL_MODELS = {
    ...
    "jsc-zixian": {
        "model": get_jsc_zixian,
        "info": MaseModelInfo(
            "jsc-s",
            model_source="physical",
            task_type="physical",
            physical_data_point_classification=True,
            is_fx_traceable=True,
        ),
    },
}
```
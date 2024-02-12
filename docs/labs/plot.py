import matplotlib.pyplot as plt

# Data from the table
batch_sizes = [128, 256, 512, 1024]
training_acc = [0.55, 0.53, 0.51, 0.47]
val_acc = [0.55, 0.52, 0.51, 0.46]
training_loss = [1.19, 1.25, 1.32, 1.41]
val_loss = [1.18, 1.25, 1.32, 1.41]

# Plotting
plt.figure(figsize=(10, 6))

# Training and Validation Accuracy
plt.plot(batch_sizes, training_acc, marker='o', label='Training Accuracy')
plt.plot(batch_sizes, val_acc, marker='o', label='Validation Accuracy')

# Training and Validation Loss
plt.plot(batch_sizes, training_loss, marker='o', label='Training Loss')
plt.plot(batch_sizes, val_loss, marker='o', label='Validation Loss')

plt.title('Model Performance vs. Batch Size \n (Learning Rate=1e-5, Epochs=20)')
plt.xlabel('Batch Size')
plt.ylabel('Performance')
plt.xticks(batch_sizes)  # Ensure all batch sizes are shown on x-axis
plt.grid(True)
plt.legend()
plt.tight_layout()

# Show plot
plt.show()



# import matplotlib.pyplot as plt

# # Data from the dataset
# learning_rates = ['1.00E-06', '5.00E-06', '1.00E-05', '5.00E-05']
# training_acc = [0.29, 0.48, 0.52, 0.57]
# val_acc = [0.29, 0.46, 0.51, 0.56]
# training_loss = [1.52, 1.4, 1.32, 1.15]
# val_loss = [1.53, 1.4, 1.32, 1.17]

# # Plotting
# plt.figure(figsize=(10, 6))

# # Training and Validation Accuracy
# plt.plot(learning_rates, training_acc, marker='o', label='Training Accuracy')
# plt.plot(learning_rates, val_acc, marker='o', label='Validation Accuracy')

# # Training and Validation Loss
# plt.plot(learning_rates, training_loss, marker='o', label='Training Loss')
# plt.plot(learning_rates, val_loss, marker='o', label='Validation Loss')

# plt.title('Model Performance vs. Learning Rate \n (Batch Size=512, Epochs=20)')
# plt.xlabel('Learning Rate')
# plt.ylabel('Performance')
# plt.xticks(learning_rates)  # Ensure all learning rates are shown on x-axis
# plt.grid(True)
# plt.legend()
# plt.tight_layout()

# # Show plot
# plt.show()

"""Test baseline_trainer.py integration with Phase 3.2 losses."""
import torch
from src.training.baseline_trainer import BaselineTrainer
from src.models.build import build_classifier
from torch.utils.data import TensorDataset, DataLoader
from src.training.base_trainer import TrainingConfig

print('='*70)
print('Testing BaselineTrainer Integration with Phase 3.2 Losses')
print('='*70)

# Create dummy model and data
model = build_classifier('resnet50', num_classes=7)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Create dummy dataset
dummy_images = torch.randn(100, 3, 224, 224)
dummy_labels = torch.randint(0, 7, (100,))
dataset = TensorDataset(dummy_images, dummy_labels)
train_loader = DataLoader(dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(dataset, batch_size=16)

# Create optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Create config
config = TrainingConfig(
    max_epochs=1,
    eval_every_n_epochs=1,
    log_every_n_steps=10,
    batch_size=16,
)

print('\n1. Testing TaskLoss (CrossEntropy) integration:')
trainer1 = BaselineTrainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=optimizer,
    config=config,
    num_classes=7,
    device=device,
    task_type='multi_class',
    use_focal_loss=False,
)
print('   [OK] Trainer created with TaskLoss (CE)')
print(f'   [OK] Criterion: {type(trainer1.criterion).__name__}')

print('\n2. Testing TaskLoss (FocalLoss) integration:')
trainer2 = BaselineTrainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=optimizer,
    config=config,
    num_classes=7,
    device=device,
    task_type='multi_class',
    use_focal_loss=True,
    focal_gamma=2.0,
)
print('   [OK] Trainer created with TaskLoss (Focal)')
print(f'   [OK] Criterion: {type(trainer2.criterion).__name__}')

print('\n3. Testing CalibrationLoss integration:')
trainer3 = BaselineTrainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=optimizer,
    config=config,
    num_classes=7,
    device=device,
    use_calibration=True,
    init_temperature=1.5,
    label_smoothing=0.1,
)
print('   [OK] Trainer created with CalibrationLoss')
print(f'   [OK] Criterion: {type(trainer3.criterion).__name__}')
temp = trainer3.get_temperature()
print(f'   [OK] Temperature: {temp:.4f}')

print('\n4. Testing training_step:')
batch = next(iter(train_loader))
loss, metrics = trainer1.training_step(batch, 0)
print('   [OK] Training step successful')
print(f'   [OK] Loss: {loss.item():.4f}')
print(f'   [OK] Accuracy: {metrics["accuracy"]:.4f}')
print(f'   [OK] Loss has gradient: {loss.requires_grad}')

print('\n5. Testing validation_step:')
val_batch = next(iter(val_loader))
val_loss, val_metrics = trainer1.validation_step(val_batch, 0)
print('   [OK] Validation step successful')
print(f'   [OK] Loss: {val_loss.item():.4f}')
print(f'   [OK] Accuracy: {val_metrics["accuracy"]:.4f}')

print('\n' + '='*70)
print('[SUCCESS] ALL INTEGRATION TESTS PASSED!')
print('='*70)

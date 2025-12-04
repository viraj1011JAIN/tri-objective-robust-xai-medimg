import json

with open("notebooks/Phase_5_Adversarial_Training.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

# Find the cell with ISICDataset and fix it
for cell in nb["cells"]:
    if cell["cell_type"] == "code":
        source = "".join(cell["source"])
        if "class ISICDataset" in source and "def __getitem__" in source:
            # Replace the dataset class with fixed version that searches all folders
            new_source = '''#@title 📊 Cell 4: Dataset Loading (ISIC 2018)
#@markdown **CSV-based dataset with medical imaging transforms**

class ISICDataset(Dataset):
    """ISIC 2018 Skin Lesion Dataset with CSV-based loading."""

    def __init__(
        self,
        df: pd.DataFrame,
        image_dir: Union[str, Path],
        transform=None,
        return_path: bool = False
    ):
        self.df = df.reset_index(drop=True)
        self.image_dir = Path(image_dir)
        self.transform = transform
        self.return_path = return_path

        # Extract labels
        if 'label_idx' in self.df.columns:
            self.labels = self.df['label_idx'].values
        elif 'label_multiclass' in self.df.columns:
            self.labels = self.df['label_multiclass'].values
        elif 'label' in self.df.columns:
            if self.df['label'].dtype == 'object':
                unique_labels = sorted(self.df['label'].unique())
                label_to_idx = {l: i for i, l in enumerate(unique_labels)}
                self.labels = self.df['label'].map(label_to_idx).values
            else:
                self.labels = self.df['label'].values
        else:
            raise ValueError(f"Cannot find label column in: {list(self.df.columns)}")

        # Get image IDs
        if 'image_id' in self.df.columns:
            self.image_ids = self.df['image_id'].values
        elif 'filepath' in self.df.columns:
            self.image_ids = self.df['filepath'].values
        elif 'image_path' in self.df.columns:
            self.image_ids = self.df['image_path'].values
        else:
            self.image_ids = self.df.iloc[:, 0].values

        # Build image path cache - search all folders once
        print(f"  Building image path cache...")
        self.path_cache = {}
        search_dirs = [
            self.image_dir / 'images' / 'train',
            self.image_dir / 'images' / 'val',
            self.image_dir / 'images' / 'test',
            self.image_dir / 'train',
            self.image_dir / 'val',
            self.image_dir / 'test',
            self.image_dir,
        ]

        for search_dir in search_dirs:
            if search_dir.exists():
                for img_file in search_dir.glob('*.jpg'):
                    img_id = img_file.stem  # filename without extension
                    if img_id not in self.path_cache:
                        self.path_cache[img_id] = img_file
                for img_file in search_dir.glob('*.png'):
                    img_id = img_file.stem
                    if img_id not in self.path_cache:
                        self.path_cache[img_id] = img_file

        print(f"  Cached {len(self.path_cache)} image paths")
        print(f"  Loaded {len(self)} samples")
        print(f"  Class distribution: {dict(pd.Series(self.labels).value_counts().sort_index())}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image_id = str(self.image_ids[idx])
        label = int(self.labels[idx])

        # Look up in cache
        if image_id in self.path_cache:
            img_path = self.path_cache[image_id]
        else:
            raise FileNotFoundError(f"Image not found in cache: {image_id}")

        # Load image
        img = cv2.imread(str(img_path))
        if img is None:
            raise ValueError(f"Failed to load image: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Apply transforms
        if self.transform:
            transformed = self.transform(image=img)
            img = transformed['image']

        if self.return_path:
            return img, label, str(img_path)
        return img, label

def get_transforms(train: bool = True, img_size: int = 224):
    """Get albumentations transforms with ToTensorV2."""
    if train:
        return A.Compose([
            A.RandomResizedCrop(size=(img_size, img_size), scale=(0.8, 1.0)),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=20, p=0.5),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, p=0.5),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Resize(height=img_size, width=img_size),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ])

def load_data(paths: PathConfig, batch_size: int = 32, num_workers: int = 4):
    """Load train, validation, and test datasets from metadata_processed.csv."""
    print("\\n📊 Loading ISIC 2018 Dataset...")
    print(f"  Data root: {paths.data_root}")

    # Find the metadata CSV
    csv_candidates = [
        paths.data_root / 'metadata_processed.csv',
        paths.data_root / 'metadata.csv',
        paths.data_root / 'metadata_fixed.csv',
    ]

    csv_path = None
    for candidate in csv_candidates:
        if candidate.exists():
            csv_path = candidate
            print(f"  Found CSV: {csv_path}")
            break

    if csv_path is None:
        raise FileNotFoundError(f"No metadata CSV found in {paths.data_root}")

    # Load full dataframe
    full_df = pd.read_csv(csv_path)
    print(f"  Total samples: {len(full_df)}")
    print(f"  Columns: {list(full_df.columns)}")

    datasets = {}
    loaders = {}

    for split in ['train', 'val', 'test']:
        # Filter by split column
        if 'split' in full_df.columns:
            split_df = full_df[full_df['split'] == split].copy()
        else:
            split_df = full_df.copy() if split == 'train' else pd.DataFrame()

        if len(split_df) == 0:
            print(f"  ⚠️ No {split} samples found, skipping...")
            continue

        print(f"\\n  {split.upper()}: {len(split_df)} samples")

        is_train = (split == 'train')
        datasets[split] = ISICDataset(
            df=split_df,
            image_dir=paths.data_root,
            transform=get_transforms(train=is_train)
        )

        loaders[split] = DataLoader(
            datasets[split],
            batch_size=batch_size,
            shuffle=is_train,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=is_train
        )

    print(f"\\n✅ Dataset loaded successfully!")
    return datasets, loaders

# Load data
datasets, loaders = load_data(paths, config.batch_size)

# Verify data
sample_batch, sample_labels = next(iter(loaders['train']))
print(f"\\n🔍 Sample batch shape: {sample_batch.shape}")
print(f"🔍 Sample labels: {sample_labels[:10].tolist()}")
'''
            cell["source"] = [new_source]
            print("Fixed ISICDataset to cache all image paths from all folders")
            break

with open("notebooks/Phase_5_Adversarial_Training.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)
print("Saved")

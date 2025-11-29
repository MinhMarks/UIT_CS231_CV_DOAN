from dataloaders.val.MapillaryDataset import MSLS

# Create dataset instance
dataset = MSLS()

print("Number of database images:", len(dataset.dbImages))
print("\nFirst 5 database images:")
print(dataset.dbImages[:5])
print("\nShape:", dataset.dbImages.shape)

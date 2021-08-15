import fiftyone as fo
import fiftyone.zoo as foz
import fiftyone.utils.openimages as oi
# Download and load the validation split of Open Images V6
print(oi.get_attributes())
dataset = foz.load_zoo_dataset("open-images-v6",
                        split="validation",
                        label_types=["classifications"],
                        max_samples=1000,
                        classes=['Airplane',
                                'Bicycle',
                                'Bird',
                                'Boat',
                                'Bottle',
                                'Bus',
                                'Car'],
                        shuffle=True,
                        only_matching=True)

session = fo.launch_app(dataset)
session.wait()
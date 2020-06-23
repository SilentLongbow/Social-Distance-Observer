## Source Code Locations

This repository is really two separate repositories purposed to work together.

The YOLOv3 detector used for this project originated from the following git repository:

https://github.com/eriklindernoren/PyTorch-YOLOv3

The deep SORT implementation, and a rough guide on the implementation are provided by the following repository and blog:

https://nanonets.com/blog/object-tracking-deepsort/#challenges?&utm_source=nanonets.com/blog/&utm_medium=blog&utm_content=DeepSORT

https://github.com/abhyantrika/nanonets_object_tracking/

## Before you Begin
You must download the YOLOv3 weights as explained in the YOLOv3 readme file.

Also, if you want to download the COCO dataset, be aware that it is approximately a 24GB download.


## Note on Code Execution

The social distance observer is run using the `run_social_distance_overseer.py` file.

If you wish to run code within the YOLOv3 directory alone, I would recommend reverting any relative imports in files such as `models.py`.

The interpreter will complain to you about them anyways.

The way to do this is to remove the '.' in front of the imported module name.

#### Arguments

Unfortunately I didn't have time to write down all the arguments and what they do here, though the arguments are explained in the main `run_social_distance_overseer.py` file.

#### Input Data

Feel free to use any videos containing people in it. I used the `.webm` files provided on the MOT challenge dataset website here:

https://motchallenge.net/

# License plate recognition

In project I'm using one template of license plate histogram to decide, which bounding box is real license plate.
When plate wasn't found, searching is repeated with different parameters. To describe signs I'm using HuMoments which
was saved as dataset. It's not the best idea, because 6 and 9 are very similar and it's almost impossible to always
describe it correctly.


Run
-
2 arguments. Path to images directory and path to output .json file

Assumptions
-
Width of plate is greater than 1/3 image width.
It's made for Polish license plates.

 Problems to fix:
 -
 Make detection independent of license plate lighting.
 Try different ways to find possible license plates.
 Improve sign description  (the 6 and 9 problem).
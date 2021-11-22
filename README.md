# SmartAnnotation
Smart Annotation project with automatic object segmentation, for lowering annotation time.

This project deploys two different annotation tools for making image annotation a semi-automated task. 
Check https://github.com/rune-l/coco-annotator for the implementation of the other annotation tool.

To run the app locally, docker is required: 
1. Download the segmentation model from here: https://drive.google.com/file/d/1iKnJnhaiVR_1OjKDaKYTIjOtcNagdWOQ/view. Save it to server folder under webapp

2. Open a cmd interface. Move to webapp folder

3. Run: docker build -t smart_annotator. 
it will take a while to build

4. Run: docker run smart_annotator

5. enjoy polygon annotations
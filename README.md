# CSE576_Dataset_Generation

Synthesize various dataset for visual-text understanding model training.

### dataset 
visual_7W: https://github.com/yukezhu/visual7w-toolkit

### usage
```python
cd CSE576_Dataset_Generation
python utils.py
```

### contribution and works 
The visual_7w is a VQA task dataset, with annotated bounding boxes for objects decribed in both texts and images.    
  
For our task, we want to convert the question-answering pairs into text desriptions for corresponding images respectively, just like 

**'who is holding the tennis racket', 'the tennis player'**  
    
 to  
  
**'the tennis player is holding the tennis racket'**   
   
and the objects here are **player** and **tennis racket**

In this part, only question type **who** is used to generate the samples.

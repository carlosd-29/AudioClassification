# Audio Classification

This is a audio classificaion model. The current classes it can classify are:

|airplane    |breathing   |brushing teeth   |can openining   |car horn   |
|---|---|---|---|---|
|cat   |chainsaw   |chirping birds   |church bells   |clapping   |
|clock alarm   |clock tick   |coughing   |cow   |crackling fire   |
|crickets   |crow   |crying baby   |dog   |door wood creaks   |


The model works by creating spectrograms, visual way of representing the signal strength, or “loudness”, of a signal over time at various frequencies present in a particular waveform, and classifying on a CNN model.

The data used is [ESC-50](https://github.com/karolpiczak/ESC-50),  a dataset wtih environmental sound classification. The dataset comes with 50 different classes but at this current stage the model is trained for 20 classes at 75% accuracy.

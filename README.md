Project PGM-OR
==========

This file contains some brief presentation of recent results of the implementation of the project

Kernel for captions (text)
--------------------------

Here some tests with the kernel distance with the image captions. 

The class CaptionCorpus has two different methods of distance between captions: one using LDA and the other using word2vec. 
Using the 100 topics for LDA and 100 dimensions for word2vec's vectors, we obtain the following results:

We search for the 3 closest images (by its captions) to this set of words: ["brown", "dog", "run", "grass", "blue", "sky", "sun"]

With LDA: 

![first](https://raw.github.com/eqperes/mvaproject/text_manip/lda_test_images/2509824208_247aca3ea3.jpg)
![second](https://raw.github.com/eqperes/mvaproject/text_manip/lda_test_images/3159447439_c1496cbaea.jpg)
![third](https://raw.github.com/eqperes/mvaproject/text_manip/lda_test_images/457875937_982588d918.jpg)

With word2vec:

![first](https://raw.github.com/eqperes/mvaproject/text_manip/w2v_test_images/3205839744_24504ba179.jpg)
![second](https://raw.github.com/eqperes/mvaproject/text_manip/w2v_test_images/1478606153_a7163bf899.jpg)
![third](https://raw.github.com/eqperes/mvaproject/text_manip/w2v_test_images/3581818450_546c89ca38.jpg)

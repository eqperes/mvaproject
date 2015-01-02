from gensim_corpus import CaptionCorpus
from getcaptions import import_captions

cap = import_captions("../../../Flickr8k_text/Flickr8k.lemma.token.txt")
cp = CaptionCorpus(cap, "English")

phrase = ["brown", "dog", "run", "grass", "blue", "sky", "sun"]

cp.ldafy(100)
im1 = cp.lda_image_caption_distance(phrase)

cp.word2vecfy(100, 1)
im2 = cp.w2v_image_caption_distance(phrase)



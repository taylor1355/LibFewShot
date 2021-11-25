# Easy data augmentation techniques for text classification
# Original Jason Wei and Kai Zou
# Updates by Shivaen Ramshetty

from eda import *
from transformer_augments import AugTransformer
import pandas as pd
import random

#arguments to be parsed from command line
import argparse
ap = argparse.ArgumentParser()
ap.add_argument("--input", required=True, type=str, help="input file of unaugmented data")
ap.add_argument("--output", required=False, type=str, help="output file of unaugmented data")
ap.add_argument("--text_id", required=False, type=str, help="id/name where the text is stored")
ap.add_argument("--num_aug", required=False, type=int, help="number of augmented sentences per original sentence")
ap.add_argument("--alpha_sr", required=False, type=float, help="percent of words in each sentence to be replaced by synonyms")
ap.add_argument("--alpha_ri", required=False, type=float, help="percent of words in each sentence to be inserted")
ap.add_argument("--alpha_rs", required=False, type=float, help="percent of words in each sentence to be swapped")
ap.add_argument("--alpha_rd", required=False, type=float, help="percent of words in each sentence to be deleted")
args = ap.parse_args()

#the output file
output = None
if args.output:
    output = args.output
else:
    from os.path import dirname, basename, join
    output = join(dirname(args.input), 'eda_' + basename(args.input))
    
text_id = 'text'
if args.text_id is not None:
    text_id = args.text_id

#number of augmented sentences to generate per original sentence
num_aug = 9 #default
if args.num_aug:
    num_aug = args.num_aug

#how much to replace each word by synonyms
alpha_sr = 0.1#default
if args.alpha_sr is not None:
    alpha_sr = args.alpha_sr

#how much to insert new words that are synonyms
alpha_ri = 0.1#default
if args.alpha_ri is not None:
    alpha_ri = args.alpha_ri

#how much to swap words
alpha_rs = 0.1#default
if args.alpha_rs is not None:
    alpha_rs = args.alpha_rs

#how much to delete words
alpha_rd = 0.1#default
if args.alpha_rd is not None:
    alpha_rd = args.alpha_rd

if alpha_sr == alpha_ri == alpha_rs == alpha_rd == 0:
     ap.error('At least one alpha should be greater than zero')

#generate more data with standard augmentation
def augment(train_orig, output_file, transformer, alpha_sr, alpha_ri, alpha_rs, alpha_rd, num_aug=9):
    
    data_df = pd.read_json(train_orig)
    out_df = pd.DataFrame()
     
    data = data_df.to_dict(orient='records')
    num = len(data)
    print("generating augmentations for", num, "sentences")
    
    i = 0
    for row in data:
        sentence = row[text_id]
        
        out_df = out_df.append(row, ignore_index=True)
        
        # eda
#         if random.randint(0,4) <= 1:
#         aug_sentences = eda(sentence, alpha_sr=alpha_sr, alpha_ri=alpha_ri, alpha_rs=alpha_rs, p_rd=alpha_rd, num_aug=num_aug)
#         for aug_sentence in aug_sentences:
#             aug_row = row
#             aug_row[text_id] = aug_sentence
#             aug_row['text'] = aug_sentence.split()
#             out_df = out_df.append(aug_row, ignore_index=True)
            
        # backtranslation
        if random.randint(0,4) == 1:
            bt_row = row
            translated = transformer.backtranslate(sentence)
            bt_row[text_id] = translated
            bt_row['text'] = translated.split()
            out_df = out_df.append(bt_row, ignore_index=True)
        
#         # gpt-2 generated
#         if random.randint(0,4) == 1:
#         gen_row = row
#         str_arr = sentence.split()
#         snippet = " ".join(str_arr[:random.randint(len(str_arr)//2, len(str_arr)-1)])
#         gen_text = transformer.generate(snippet, random.randint(10, 20))
#         gen_row[text_id] = gen_text
#         gen_row['text'] = gen_text.split()
#         out_df = out_df.append(gen_row, ignore_index=True)
    
        i += 1
        if i % 1000 == 0:
            print(i)
            out_df.to_json(f'amazon_backtranslate_{i}_train.json', orient='records')
    
    out_df.to_json(output_file, orient='records')
    print("generated augmented sentences for " + train_orig + " to " + output_file)

#main function
if __name__ == "__main__":
    
    aug_transformer = AugTransformer()
    #generate augmented sentences and output into a new file
    augment(args.input, output, aug_transformer, alpha_sr=alpha_sr, alpha_ri=alpha_ri, alpha_rs=alpha_rs, alpha_rd=alpha_rd, num_aug=num_aug)
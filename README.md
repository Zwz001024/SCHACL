SCHACL:Improving short text clustering through hybrid augmentation and contrastive learning 

To acknowledge and respect the authors who first applied contrastive learning to short text clustering tasks, we cite their work as follows:
Baseline paper# SCCL: Supporting Clustering with Contrastive Learning
## Citation:
```bibtex
@inproceedings{zhang-etal-2021-supporting,
    title = "Supporting Clustering with Contrastive Learning",
    author = "Zhang, Dejiao  and
      Nan, Feng  and
      Wei, Xiaokai  and
      Li, Shang-Wen  and
      Zhu, Henghui  and
      McKeown, Kathleen  and
      Nallapati, Ramesh  and
      Arnold, Andrew O.  and
      Xiang, Bing",
    booktitle = "Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies",
    month = jun,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.naacl-main.427",
    doi = "10.18653/v1/2021.naacl-main.427",
    pages = "5419--5430",
```

### Dependencies:
    python==3.6.13 
    pytorch==1.6.0. 
    sentence-transformers==2.0.0. 
    transformers==4.8.1. 
    tensorboardX==2.4.1
    pandas==1.1.5
    sklearn==0.24.1
    numpy==1.19.5


"SCHACL, similar to SCCL, employs two types of data augmentation: explicit augmentations and virtual augmentations. The choice of augmentation method directly impacts the performance of subsequent clustering tasks.

Given the significant influence of this paper, many researchers have attempted to reproduce SCCL. However, inconsistent reproduction results have been reported across multiple studies focusing on short text clustering.

Therefore, it is worth noting that the data augmentation option based on explicit augmentations tends to yield relatively better clustering performance.
"

### explicit augmentations (parser.add_argument('--augtype', type=str, default='explicit', choices=['virtual', 'explicit']))
Step-1. download the original datastes from https://github.com/rashadulrakib/short-text-clustering-enhancement/tree/master/data

step-2. then obtain the augmented data using the code in ./AugData/

step-3  SCHACL requires the use of distinct data augmentation techniques on each original sample to generate a set of augmented data.
The data format is (label, text, text1, text2, text3), where text is the original data, and text1, text2, and text3 represent different augmented data instances. 

step-4
Regarding the backbone network used, i.e., the pretrained model, you need to access Hugging Face via a valid network configuration to manually download distilbert-base-nli-stsb-mean-tokens.
https://huggingface.co/sentence-transformers/distilbert-base-nli-mean-tokens
https://huggingface.co/roberta-large, 
https://huggingface.co/bert-base-uncased

step-5  Please place the processed dataset in the following directory.  
        parser.add_argument('--datapath', type=str, default='data')
        Modify your parameter settings and then run main.py.

python main.py \
        --resdir $path-to-store-your-results \
        --use_pretrain SBERT \  (optimizer.py)
        --bert distilbert \
        --datapath $path-to-your-data \
        --dataname  \
        --num_classes \
        --augtype explicit \
        --temperature 0.5 \
        --eta 10 \
        --lr 1e-05 \
        --lr_scale 100 \
        --max_length 32 \
        --batch_size 600 \
        --max_iter 5000 \
        --print_freq 100 \
        --gpuid 0 &

The max_length parameter needs to be adjusted based on the dataset. You can refer to the text length of the dataset: if some text data in the dataset is too long, you should increase the max_length setting. However, you also need to take your GPU memory into account.



Based on my personal computer and runtime environment, the following are the optimal parameter settings on the best-performing datasets for reference.
S  -c-d-t0.1-
Max length==45          batch_size', default=600
alpha', type=float, default=1.   eta=3

tweet  -c-d-t0.2-
Max length==32          batch_size', default=600
alpha', type=float, default=1.   eta=3

T  -a-d-t0.1-
Max length==42          batch_size', default=600
alpha', type=float, default=1.   eta=3

TS  -a-d-t0.2-
Max length==42          batch_size', default=600
alpha', type=float, default=1.   eta=1

biomedical -a-d-t-
Max length==42          batch_size', default=600
alpha', type=float, default=10.   eta=10

stack  -a-d-t-
Max length==42          batch_size', default=600
alpha', type=float, default=10.   eta=10

sea  -c-d-t-
Max length==42          batch_size', default=600
alpha', type=float, default=1.   eta=10

agnews  -a-d-t0.2-
Max length==45          batch_size', default=600
alpha', type=float, default=1.   eta=1

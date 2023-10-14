Introducing CAD: the Contextual Abuse Dataset Bertie Vidgen, Dong Nguyen, Helen Margetts, 
Patricia Rossini, Rebekah Tromble, NAACL 2021

Online abuse can inflict harm on users and communities, making online spaces unsafe and toxic. 
Progress in automatically detecting and classifying abusive content is often held back by the 
lack of high quality and detailed datasets. We introduce a new dataset of primarily English 
Reddit entries which addresses several limitations of prior work. It (1) contains six conceptually 
distinct primary categories as well as secondary categories, (2) has labels annotated in the 
context of the conversation thread, (3) contains rationales and (4) uses an expert-driven 
group-adjudication process for high quality annotations. This repository contains the annotated 
dataset, annotation guidelines and the trained models and their output.

===============================================================================================

CODE: https://github.com/dongpng/cad_naacl2021

PAPER: https://www.aclweb.org/anthology/2021.naacl-main.182/

CITATION:

@inproceedings{vidgen-etal-2021-introducing,
    title = "Introducing {CAD}: the Contextual Abuse Dataset",
    author = "Vidgen, Bertie  and
      Nguyen, Dong  and
      Margetts, Helen  and
      Rossini, Patricia  and
      Tromble, Rebekah",
    booktitle = "Proceedings of the 2021 Conference of the North American Chapter of the 
                Association for Computational Linguistics: Human Language Technologies",
    month = jun,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2021.naacl-main.182",
    pages = "2289--2303",
    abstract = "Online abuse can inflict harm on users and communities, making online 
                spaces unsafe and toxic. Progress in automatically detecting and classifying 
                abusive content is often held back by the lack of high quality and detailed 
                datasets.We introduce a new dataset of primarily English Reddit entries which 
                addresses several limitations of prior work. It (1) contains six conceptually 
                distinct primary categories as well as secondary categories, (2) has labels 
                annotated in the context of the conversation thread, (3) contains rationales 
                and (4) uses an expert-driven group-adjudication process for high quality 
                annotations. We report several baseline models to benchmark the work of future 
                researchers. The annotated dataset, annotation guidelines, models and code 
                are freely available.",
}

DOI: 10.5281/zenodo.4881008

===============================================================================================

VERSIONS 1.0 and 1.1

Note about the dataset (v1 vs. v1.1) cad_v1 was used to produce the results in the NAACL 
2021 paper. We identified some minor issues later. This affects the primary and secondary 
categories of 95 entries. The new version CAD v1.1 is also provided, based on the changes 
recorded in data/errata_v1_to_v1_1.

===============================================================================================

DIRECTORIES

- Data (dataset, codebook, etc.)
- Experiments (trained models and their output, see the Github repository)

===============================================================================================

CONTACT

Questions or comments about the data? Contact Bertie Vidgen.




# MMM2021 - Graph Index for Lifelog Moment Retrieval
There are 4 main steps
1. Generate Scene Graph (SG) for lifelog images
2. Embed each SG into 2 matrices
3. Parse a query to a semantic graph
4. Retrieval (embed semantic graph and compare with embedded SG to rank the result)

The jupyter notebook files are just for the examples on how it works and the results. The queries and the result experiments are stored in **Queries_Bank folder**

## Generate Scene Graph
This step is totally based on the original repository of [KaihuaTang](https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch). Please follow his install instruction. As the time we did this project, KaihuaTang had not released the code for generating a SG for a custom image (but now he did). Therefore we changed a little bit things in his code to make it run. Also we add the option to filter out predictions with low score. 

There is a jupyter file to test the SGG on a custom image. You need to download the pretrain model SGDet from [KaihuaTang](https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch) and put it in pretrain folder. The `generate_sgg_lsc.py` file to generate all SG for lifelog images.

## Embedding Scene Graph
This step is in **Embedding-SG folder**

This step processes SG detected on the 1st step (for example: get fully connected graphs, or maximum spanning tree)

## Generate Semantic Graph From A Query
This step is in **Parsing-Query folder**

We used the repository of [vacancy](https://github.com/vacancy/SceneGraphParser) to parse the text into a graph with the help of rule-based approach.

Prior to this, we need to extract the information of date and location from the query. We utilized the funtions from MySceal system.

## Retrieval
In the **Retrieval folder**. Note that we will use all function defined in other folders as well.

The `extract_mysceal_data.py` is used to extract other information of a lifelog image (date, location, concept, ...) that used in MySceal system.


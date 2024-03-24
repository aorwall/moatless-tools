# Benchmark summary

* Dataset: princeton-nlp/SWE-bench_Lite:dev
* Embedding model: text-embedding-3-small
* Splitter: `EpicSplitter(chunk_size=750, min_chunk_size=100, comment_strategy=CommentStrategy.ASSOCIATE)`
* Retrieve strategy: `RetrieveStrategy.FILES`

## Instances

| instance_id | no_of_patches | any_found_context_length | all_found_context_length | avg_pos | min_pos | max_pos | top_file_pos | missing_snippets | missing_patch_files |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [marshmallow-code__marshmallow-1343](marshmallow-code__marshmallow-1343/report.md) | 1 | 8315 | 8315 | 2.0 | 1 | 1 | 1 | 2 | 0 |
| [marshmallow-code__marshmallow-1359](marshmallow-code__marshmallow-1359/report.md) | 1 | 22589 | 22589 | 3.0 | 3 | 3 | 3 | 1 | 0 |
| [pvlib__pvlib-python-1072](pvlib__pvlib-python-1072/report.md) | 1 | 6866 | 6866 | 1.0 | 1 | 1 | 1 | 1 | 0 |
| [pvlib__pvlib-python-1154](pvlib__pvlib-python-1154/report.md) | 1 | 34804 | 34804 | 1.0 | 1 | 1 | 1 | 1 | 0 |
| [pvlib__pvlib-python-1606](pvlib__pvlib-python-1606/report.md) | 1 |  |  | 0.0 |  |  |  | 3 | 1 |
| [pvlib__pvlib-python-1707](pvlib__pvlib-python-1707/report.md) | 1 | 8386 | 8386 | 2.0 | 1 | 1 | 1 | 2 | 0 |
| [pvlib__pvlib-python-1854](pvlib__pvlib-python-1854/report.md) | 1 | 26069 | 26069 | 2.0 | 1 | 1 | 1 | 2 | 0 |
| [pydicom__pydicom-1139](pydicom__pydicom-1139/report.md) | 1 | 8394 | 8394 | 4.0 | 2 | 2 | 2 | 2 | 0 |
| [pydicom__pydicom-1256](pydicom__pydicom-1256/report.md) | 1 | 8637 | 8637 | 2.0 | 2 | 2 | 2 | 1 | 0 |
| [pydicom__pydicom-1413](pydicom__pydicom-1413/report.md) | 1 | 7088 | 7088 | 1.0 | 1 | 1 | 1 | 1 | 0 |
| [pydicom__pydicom-1694](pydicom__pydicom-1694/report.md) | 1 | 28025 | 28025 | 2.0 | 2 | 2 | 2 | 1 | 0 |
| [pydicom__pydicom-901](pydicom__pydicom-901/report.md) | 1 | 968 |  | 6.0 | 1 | 2 | 1 | 3 | 0 |
| [pylint-dev__astroid-1196](pylint-dev__astroid-1196/report.md) | 1 | 42052 | 42052 | 2.0 | 2 | 2 | 2 | 1 | 0 |
| [pylint-dev__astroid-1268](pylint-dev__astroid-1268/report.md) | 1 | 5981 | 5981 | 6.0 | 1 | 2 | 1 | 2 | 0 |
| [pylint-dev__astroid-1333](pylint-dev__astroid-1333/report.md) | 1 | 6194 | 6194 | 4.0 | 2 | 2 | 2 | 2 | 0 |
| [pylint-dev__astroid-1866](pylint-dev__astroid-1866/report.md) | 1 | 17853 | 17853 | 5.0 | 5 | 5 | 3 | 1 | 0 |
| [pylint-dev__astroid-1978](pylint-dev__astroid-1978/report.md) | 1 | 45004 | 45004 | 30.0 | 10 | 10 | 9 | 3 | 0 |
| [pyvista__pyvista-4315](pyvista__pyvista-4315/report.md) | 1 | 6893 | 6893 | 1.0 | 1 | 1 | 1 | 1 | 0 |
| [sqlfluff__sqlfluff-1517](sqlfluff__sqlfluff-1517/report.md) | 1 |  |  | 0.0 |  |  |  | 2 | 1 |
| [sqlfluff__sqlfluff-1625](sqlfluff__sqlfluff-1625/report.md) | 1 | 1498 | 1498 | 11.0 | 1 | 8 | 1 | 1 | 0 |
| [sqlfluff__sqlfluff-1733](sqlfluff__sqlfluff-1733/report.md) | 1 |  |  | 0.0 |  |  |  | 1 | 1 |
| [sqlfluff__sqlfluff-1763](sqlfluff__sqlfluff-1763/report.md) | 1 |  |  | 0.0 |  |  |  | 2 | 1 |
| [sqlfluff__sqlfluff-2419](sqlfluff__sqlfluff-2419/report.md) | 1 |  |  | 0.0 |  |  | 1 | 1 | 0 |

# Recall

|     | 13k | 27k | 50k |
| --- | --- | --- | --- |
| All | 47.83% | 60.87% | 78.26% |

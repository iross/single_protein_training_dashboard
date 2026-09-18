uv run mldag-query db build --checkpoint-dir /staging/i/iaross/single_protein_models_gpu_device_constrained --db provenance.db
uv run mldag-query db build --checkpoint-dir /staging/i/iaross/single_protein_checkpoints_with_ospool --db mixed.db
uv run mldag-query db build --checkpoint-dir /staging/i/iaross/single_protein_models_dgxspark --db dgxspark.db

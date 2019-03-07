#!/bin/bash

problem_dir="starting_kit/ingestion_program"
input_dir="starting_kit/public_data"
model_dir="starting_kit/example_submission"

python ${problem_dir}/ingestion.py ${input_dir} ${input_dir}/res ${problem_dir} ${model_dir}

score_dir="starting_kit/scoring_program"
scoring_output_dir="starting_kit/output_dir"

python ${score_dir}/evaluate.py ${input_dir} ${scoring_output_dir}

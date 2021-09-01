##nb: copied from ag1000g/jobs/create_training_data.080921.job
##run as an array of ten jobs with 8G memory each

control_dir=/proj/dschridelab/rrlove/ag1000g/data/ms/ms_modified/training/control_files/
base_dir=/proj/dschridelab/rrlove/ag1000g/data/ms/ms_modified/training/

##full model, two-way introgression

cd ${base_dir}output/two_way_introgression/out_080921/
echo "starting full model"

for i in {1..10}; do \
mkdir out_080921_${SLURM_ARRAY_TASK_ID}_${i} && cd out_080921_${SLURM_ARRAY_TASK_ID}_${i};
~/bin/ArchIE/msmodified/ms_recompile 306 250 -t tbs -r tbs 10000 \
-I 2 150 156 \
-n 2 tbs \
-en tbs 1 tbs \
-en tbs 1 tbs \
-ej tbs 2 1 \
-en tbs 1 tbs \
-en tbs 2 tbs \
-ej tbs 3 2 \
-ej tbs 4 1 \
-en tbs 4 tbs \
-es tbs 2 tbs \
-es tbs 1 tbs < ${control_dir}two_way_introgression_expanded.txt > out_080921_${SLURM_ARRAY_TASK_ID}_${i}.txt;
cd ..;
done

##introgression from BF to AO

cd ${base_dir}output/BF_to_AO/out_080921/
echo "starting BF-to-AO introgression"

for i in {1..10}; do \
mkdir out_080921_${SLURM_ARRAY_TASK_ID}_${i} && cd out_080921_${SLURM_ARRAY_TASK_ID}_${i};
~/bin/ArchIE/msmodified/ms_recompile 306 250 -t tbs -r tbs 10000 \
-I 2 150 156 \
-n 2 tbs \
-en tbs 1 tbs \
-en tbs 1 tbs \
-ej tbs 2 1 \
-en tbs 1 tbs \
-en tbs 2 tbs \
-ej tbs 3 1 \
-en tbs 3 tbs \
-es tbs 2 tbs < ${control_dir}BF_to_AO_expanded.txt > out_080921_${SLURM_ARRAY_TASK_ID}_${i}.txt;
cd ..;
done

##introgression from AO to BF

cd ${base_dir}output/AO_to_BF/out_080921/
echo "starting AO-to-BF introgression"

for i in {1..10}; do \
mkdir out_080921_${SLURM_ARRAY_TASK_ID}_${i} && cd out_080921_${SLURM_ARRAY_TASK_ID}_${i};
~/bin/ArchIE/msmodified/ms_recompile 306 250 -t tbs -r tbs 10000 \
-I 2 150 156 \
-n 2 tbs \
-en tbs 1 tbs \
-en tbs 1 tbs \
-ej tbs 2 1 \
-en tbs 1 tbs \
-en tbs 2 tbs \
-ej tbs 3 2 \
-es tbs 1 tbs < ${control_dir}AO_to_BF_expanded.txt > out_080921_${SLURM_ARRAY_TASK_ID}_${i}.txt;
cd ..;
done

##no introgression

cd ${base_dir}output/no_introgression/out_080921/
echo "starting no introgression"

for i in {1..10}; do \
mkdir out_080921_${SLURM_ARRAY_TASK_ID}_${i} && cd out_080921_${SLURM_ARRAY_TASK_ID}_${i};
~/bin/ArchIE/msmodified/ms_recompile 306 250 -t tbs -r tbs 10000 \
-I 2 150 156 \
-n 2 tbs \
-en tbs 1 tbs \
-en tbs 1 tbs \
-ej tbs 2 1 \
-en tbs 1 tbs \
-en tbs 2 tbs \
-ej tbs 3 2 \
-es tbs 1 0.9999999999999 < ${control_dir}no_introgression_expanded.txt > out_080921_${SLURM_ARRAY_TASK_ID}_${i}.txt;
cd ..;
done

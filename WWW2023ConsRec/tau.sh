python -u main.py --dataset=Mafengwo --predictor=MLP --loss_type=BPR --learning_rate=0.0001 --device=cuda:0 --num_negatives=8 --layers=3 --epoch=200 --tau=1

python -u main.py --dataset=Mafengwo --predictor=MLP --loss_type=BPR --learning_rate=0.0001 --device=cuda:0 --num_negatives=8 --layers=3 --epoch=200 --tau=2

python -u main.py --dataset=Mafengwo --predictor=MLP --loss_type=BPR --learning_rate=0.0001 --device=cuda:0 --num_negatives=8 --layers=3 --epoch=200 --tau=4

python -u main.py --dataset=Mafengwo --predictor=MLP --loss_type=BPR --learning_rate=0.0001 --device=cuda:0 --num_negatives=8 --layers=3 --epoch=200 --tau=8

python -u main.py --dataset=Mafengwo --predictor=MLP --loss_type=BPR --learning_rate=0.0001 --device=cuda:0 --num_negatives=8 --layers=3 --epoch=200 --tau=16

python -u main.py --dataset=Mafengwo --predictor=MLP --loss_type=BPR --learning_rate=0.0001 --device=cuda:0 --num_negatives=8 --layers=3 --epoch=200 --tau=32

python -u main.py --dataset=Mafengwo --predictor=MLP --loss_type=BPR --learning_rate=0.0001 --device=cuda:0 --num_negatives=8 --layers=3 --epoch=200 --tau=64

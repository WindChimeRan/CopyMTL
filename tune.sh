python main.py --gpu 0 --cell lstm --decoder_type one > ./log/lstm_one.txt &
python main.py --gpu 1 --cell lstm --decoder_type multi > ./log/lstm_multi.txt &
python main.py --gpu 2 --cell gru --decoder_type multi > ./log/gru_multi.txt &
python main.py --gpu 3 --cell gru --decoder_type one > ./log/gru_one.txt &

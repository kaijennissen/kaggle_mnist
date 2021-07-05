for model in bayesian deterministic probabilistic;
do
    python mnist_mwe.py --model=$model --predict
done;

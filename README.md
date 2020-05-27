# uncertaintyASR

## build docker

run the following command:

    $ docker build -t uncertainty_docker .


## run docker
    
    $ docker run --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=0 \
        --rm \
        -v <path-to-repo>/src:/root/asr-python/src \
        -v <path-to-repo>/exp:/root/asr-python/exp \
        -v <path-to-repo>/results:/root/asr-python/results \
        -v <path-to-dataset>/TIDIGITS-ASE:/root/asr-python/TIDIGITS-ASE \
        -it uncertainty_docker \
        python3 /root/asr-python/src/recognizer_torch.py 'NN'
    
 Depending on the model use 'NN', 'dropout', 'BNN2', or 'ensemble'
 
 <path-to-dataset> must at least contain the wav files for which we want to create adverarial examples.
    
 ## run eval
 
 After calculating the adversarial examples, the evaluation on the uncertainty features can be called via:
 
     docker run --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=0 \
            --rm \
            -v <path-to-repo>/src:/root/asr-python/src \
            -v <path-to-repo>/exp:/root/asr-python/exp \
            -v <path-to-repo>/results:/root/asr-python/results \
            -v <path-to-dataset>/TIDIGITS-ASE:/root/asr-python/TIDIGITS-ASE \
            -it uncertainty_docker \
            python3 /root/asr-python/src/eval.py

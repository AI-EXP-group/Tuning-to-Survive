

start_idx=0
end_idx=3

device=cuda:0

# cifar10

# train clean model
for((i=start_idx; i<end_idx; i++))
do
    echo "python train_clean.py $i"
    python train_clean.py --idx $i --dataset cifar10 --model res18 --device $device
done

# get trigger
for((i=start_idx; i<end_idx; i++))
do
    echo "python get_trigger.py $i"
    python get_trigger.py --idx $i --dataset cifar10 --model res18 --device $device
done

# watermarking
for((i=start_idx; i<end_idx; i++))
do
    echo "python watermarking.py $i"
    python watermarking.py --idx $i --dataset cifar10 --model res18 --device $device
done

# t2s
for((i=start_idx; i<end_idx; i++))
do
    echo "t2s $i"
    python t2s.py --idx $i --dataset cifar10 --model res18 --alpha 50 --device $device
done


# ------------------------------------------------------------------------------------------------------------


# extraction soft label
for((i=start_idx; i<end_idx; i++))
do
    echo "extraction soft label $i"
    python extraction.py --idx $i --target_dataset cifar10 --stolen_model res18 --device $device
done    

# extraction hard label
for((i=start_idx; i<end_idx; i++))
do
    echo "extraction hard label $i"
    python extraction.py --idx $i --target_dataset cifar10 --stolen_model res18 --hard_label --device $device
done

# extraction with different dataset
for((i=start_idx; i<end_idx; i++))
do
    echo "extraction with different dataset $i"
    python extraction.py --idx $i --target_dataset cifar10 --sur_dataset stl10 --device $device
    python extraction.py --idx $i --target_dataset cifar10 --sur_dataset cifar100 --device $device
    python extraction.py --idx $i --target_dataset cifar10 --sur_dataset tinyimagenet --device $device
done    

# extraction with different model
for((i=start_idx; i<end_idx; i++))
do
    echo "extraction with different model $i"
    python extraction.py --idx $i --target_dataset cifar10 --stolen_model wrn --device $device
    python extraction.py --idx $i --target_dataset cifar10 --stolen_model dense121 --device $device
    python extraction.py --idx $i --target_dataset cifar10 --stolen_model googlenet --device $device
done

# double extraction
for((i=start_idx; i<end_idx; i++))
do
    echo "double extraction $i"
    python extraction.py --idx $i --target_dataset cifar10 --stolen_model res18 --double_extraction --device $device
done

# double extraction with different dataset
for((i=start_idx; i<end_idx; i++))
do
    echo "double extraction with different dataset $i"
    python extraction.py --idx $i --target_dataset cifar10 --stolen_model res18 --sur_dataset stl10 --double_extraction --device $device
done





# ------------------------------------------------------------------------------------------------------------


# pruning
for ((i=start_idx; i<end_idx; i++))
do
    for ((j=0; j<11; j++))
    do
        sparsity=$(echo "scale=1; $j/10" | bc)
        echo "python pruning.py $i $sparsity"
        python pruning.py --idx $i --sparsity $sparsity --dataset cifar10
    done
done

# quantization
for ((i=start_idx; i<end_idx; i++))
do
    for j in 4 8 16
    do
        bits=$j
        echo "python quantization.py $i $bits"
        python quantization.py --idx "$i" --bits "$bits" --dataset cifar10
    done
done

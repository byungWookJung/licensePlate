
## train_advanced_digit.py
python train_advanced_digit.py -s output -de output/adv_digitetc.cpickle -k output/adv_hangul.cpickle -c output/adv_char.cpickle -d output/adv_digit.cpickle

## recognize.py
python recognize.py --images ../testing_lp_dataset --digitetc-classifier output/adv_digitetc.cpickle --korchar-classifier output/adv_hangul.cpickle --char-classifier output/adv_char.cpickle --digit-classifier output/adv_digit.cpickle
python recognize.py -i ../testing_lp_dataset -de output/adv_digitetc.cpickle -k output/adv_hangul.cpickle -c output/adv_char.cpickle -d output/adv_digit.cpickle
python recognize.py -i ../kor_lp_data -de output/adv_digitetc.cpickle -k output/adv_hangul.cpickle -c output/adv_char.cpickle -d output/adv_digit.cpickle
python recognize.py -i autoever -de output/adv_digitetc.cpickle -k output/adv_hangul.cpickle -c output/adv_char.cpickle -d output/adv_digit.cpickle
python recognize.py -i ../kor_lp_data -de output/adv_digitetc.cpickle -k output/adv_hangul.cpickle -c output/adv_char.cpickle -d output/adv_digit.cpickle
python recognize.py -i now_test -de output/adv_digitetc.cpickle -k output/adv_hangul.cpickle -c output/adv_char.cpickle -d output/adv_digit.cpickle

all:
	rnn_prof irt assistments skill_builder_data_big.txt --onepo \
	--drop-duplicates --no-remove-skill-nans --num-folds 5 \
	--item-id-col problem_id --concept-id-col single 

assist09:
	rnn_prof irt assist09 skill_builder_data_big.txt --onepo \
	--drop-duplicates --no-remove-skill-nans --num-folds 5 \
	--item-id-col problem_id --concept-id-col single >> assist09-simple.txt 2> assist09-output.txt

assist12:
	rnn_prof irt assist12 skill_builder_data_big.txt --onepo \
	--drop-duplicates --no-remove-skill-nans --num-folds 5 \
	--item-id-col problem_id --concept-id-col single >> assist12-simple.txt 2> assist12-output.txt

assist15:
	rnn_prof irt assist15 skill_builder_data_big.txt --onepo \
	--drop-duplicates --no-remove-skill-nans --num-folds 5 \
	--item-id-col problem_id --concept-id-col single >> assist15-simple.txt 2> assist15-output.txt

kdd:
	rnn_prof irt kdd skill_builder_data_big.txt --onepo \
	--drop-duplicates --no-remove-skill-nans --num-folds 5 \
	--item-id-col problem_id --concept-id-col single >> kdd-simple.txt 2> kdd-output.txt

movie100k:
	rnn_prof irt movie100k-1fold skill_builder_data_big.txt --onepo \
	--drop-duplicates --no-remove-skill-nans --num-folds 1 \
	--item-id-col problem_id --concept-id-col single >> movie100k-1fold-simple.txt 2> movie100k-1fold-output.txt

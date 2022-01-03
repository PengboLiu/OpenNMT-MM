python translate.py -model save_model/modal_1229_step_120000.pt \
-src data/test_2017_mscoco.lc.norm.tok.bpe.en -output pred_120000_beam5_2017_mscoco.txt \
-gpu 7  -beam_size 5

sed -r 's/(@@ )|(@@ ?$)//g' pred_120000_beam5_2017_mscoco.txt> pred_120000_beam5_2017_mscoco.tok.txt
rm pred_120000_beam5_2017_mscoco.txt

python translate.py -model save_model/modal_1229_step_121000.pt \
-src data/test_2017_mscoco.lc.norm.tok.bpe.en -output pred_121000_beam5_2017_mscoco.txt \
-gpu 7  -beam_size 5

sed -r 's/(@@ )|(@@ ?$)//g' pred_121000_beam5_2017_mscoco.txt> pred_121000_beam5_2017_mscoco.tok.txt
rm pred_121000_beam5_2017_mscoco.txt

python translate.py -model save_model/modal_1229_step_122000.pt \
-src data/test_2017_mscoco.lc.norm.tok.bpe.en -output pred_122000_beam5_2017_mscoco.txt \
-gpu 7  -beam_size 5

sed -r 's/(@@ )|(@@ ?$)//g' pred_122000_beam5_2017_mscoco.txt> pred_122000_beam5_2017_mscoco.tok.txt
rm pred_122000_beam5_2017_mscoco.txt

python translate.py -model save_model/modal_1229_step_123000.pt \
-src data/test_2017_mscoco.lc.norm.tok.bpe.en -output pred_123000_beam5_2017_mscoco.txt \
-gpu 7  -beam_size 5

sed -r 's/(@@ )|(@@ ?$)//g' pred_123000_beam5_2017_mscoco.txt> pred_123000_beam5_2017_mscoco.tok.txt
rm pred_123000_beam5_2017_mscoco.txt

python translate.py -model save_model/modal_1229_step_124000.pt \
-src data/test_2017_mscoco.lc.norm.tok.bpe.en -output pred_124000_beam5_2017_mscoco.txt \
-gpu 7  -beam_size 5

sed -r 's/(@@ )|(@@ ?$)//g' pred_124000_beam5_2017_mscoco.txt> pred_124000_beam5_2017_mscoco.tok.txt
rm pred_124000_beam5_2017_mscoco.txt

python translate.py -model save_model/modal_1229_step_125000.pt \
-src data/test_2017_mscoco.lc.norm.tok.bpe.en -output pred_125000_beam5_2017_mscoco.txt \
-gpu 7  -beam_size 5

sed -r 's/(@@ )|(@@ ?$)//g' pred_125000_beam5_2017_mscoco.txt> pred_125000_beam5_2017_mscoco.tok.txt
rm pred_125000_beam5_2017_mscoco.txt

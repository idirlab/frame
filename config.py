from Hydra.utils.english_spelling import us2gb, gb2us
import os

root_path = os.path.dirname(os.path.abspath(__file__))

bert_version = 'roberta-base'

raw_path = f'{root_path}/data/raw'
framenet_path = f'{root_path}/data/raw/fndata-1.7'
processed_path = f'{root_path}/data/processed'
cache_dir = f'{root_path}/cache'
logs_dir = f'{root_path}/logs'
model_dir = f'{root_path}/model_outputs'


OPENSESAME_TEST_FILES = [
        "ANC__110CYL067.xml",
        "ANC__110CYL069.xml",
        "ANC__112C-L013.xml",
        "ANC__IntroHongKong.xml",
        "ANC__StephanopoulosCrimes.xml",
        "ANC__WhereToHongKong.xml",
        "KBEval__atm.xml",
        "KBEval__Brandeis.xml",
        "KBEval__cycorp.xml",
        "KBEval__parc.xml",
        "KBEval__Stanford.xml",
        "KBEval__utd-icsi.xml",
        "LUCorpus-v0.3__20000410_nyt-NEW.xml",
        "LUCorpus-v0.3__AFGP-2002-602187-Trans.xml",
        "LUCorpus-v0.3__enron-thread-159550.xml",
        "LUCorpus-v0.3__IZ-060316-01-Trans-1.xml",
        "LUCorpus-v0.3__SNO-525.xml",
        "LUCorpus-v0.3__sw2025-ms98-a-trans.ascii-1-NEW.xml",
        "Miscellaneous__Hound-Ch14.xml",
        "Miscellaneous__SadatAssassination.xml",
        "NTI__NorthKorea_Introduction.xml",
        "NTI__Syria_NuclearOverview.xml",
        "PropBank__AetnaLifeAndCasualty.xml",
        ]

OPENSESAME_DEV_FILES = [
        "ANC__110CYL072.xml",
        "KBEval__MIT.xml",
        "LUCorpus-v0.3__20000415_apw_eng-NEW.xml",
        "LUCorpus-v0.3__ENRON-pearson-email-25jul02.xml",
        "Miscellaneous__Hijack.xml",
        "NTI__NorthKorea_NuclearOverview.xml",
        "NTI__WMDNews_062606.xml",
        "PropBank__TicketSplitting.xml",
        ]

OPENSESAME_TRAIN_FILES = list(set(os.listdir(f'{framenet_path}/fulltext')) - set(OPENSESAME_DEV_FILES) - set(OPENSESAME_TEST_FILES))

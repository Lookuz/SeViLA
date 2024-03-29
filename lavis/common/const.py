# SeViLA training tasks
SEVILA_REFINE_LOCALIZER = "train_localizer"
SEVILA_FINETUNE_ANSWERER = "train_answerer"

# SeViLA prompts
SEVILA_ANSWERER_PROMPT_POSTFIX = "Considering the information presented in the frame, select the correct answer from the options."
SEVILA_LOCALIZER_PROMPT_POSTFIX = "Does the information within the frame provide the necessary details to accurately answer the given question?"
CONVILA_FRAME_PREDICTION_PROMPT_POSTFIX = "Given shuffled visual frames numbered {} to {}, a question over the frames and the correct answer to the question, predict the original order of the frames."

# SeViLA model hyperparameters
SEVILA_MAX_TEXT_LENGTH = 77
SEVILA_ANSWER_IDS = [71, 272, 205, 309, 262] # A B C D E
SEVILA_ANSWER_MAP = {'A' : 0, 'B' : 1, 'C' : 2, 'D' : 3, 'E' : 4}
SEVILA_FRAME_PREFIX = ["Frame: "]
SEVILA_MULTI_FRAME_PREFIX = "Frame {}:"
SEVILA_PSEUDO_LABEL_POSITIVE = "yes"
SEVILA_ID_POSITIVE = 4273
SEVILA_PSEUDO_LABEL_NEGATIVE = "no"
SEVILA_ID_NEGATIVE = 150

# ConViLA tasks
CONVILA_TRAIN_E2E = "end-to-end"
CONVILA_TRAIN_ALTERNATE = "alternate"
CONVILA_TRAIN_MULTITASK = "multitask"
CONVILA_TRAIN_REFINEPLUS = "refine-plus"
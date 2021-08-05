import torch

from transformers import(
    AutoModelForSeq2SeqLM, 
    PreTrainedModel,
    PreTrainedTokenizer,
    AutoConfig,
    T5Tokenizer
)

LANGUAGE_PORTUGUES = "pt_br"
LANGUAGE_ENGLISH = "en"

class E2EQGPipeline:
    def __init__(
        self,
        model_path_or_name: str,
        tokenizer_path_or_name: str,
        use_cuda: bool,
        language:str = LANGUAGE_PORTUGUES
    ) :
        self.language = language

        self.model:PreTrainedModel = AutoModelForSeq2SeqLM.from_pretrained(model_path_or_name)
        
        self.tokenizer:PreTrainedTokenizer = T5Tokenizer.from_pretrained(tokenizer_path_or_name, config=AutoConfig.from_pretrained(model_path_or_name))

        self.device = "cuda" if torch.cuda.is_available() and use_cuda else "cpu"
        self.model.to(self.device)

        assert self.model.__class__.__name__ in ["T5ForConditionalGeneration"]
        
        self.model_type = "t5"
        
        self.default_generate_kwargs = {
            "max_length": 256,
            "num_beams": 4,
            "length_penalty": 1.5,
            "no_repeat_ngram_size": 3,
            "early_stopping": True,
        }
    
    def __call__(self, context: str, **generate_kwargs):
        inputs = self._prepare_inputs_for_e2e_qg(context)

        # TODO: when overrding default_generate_kwargs all other arguments need to be passsed
        # find a better way to do this
        if not generate_kwargs:
            generate_kwargs = self.default_generate_kwargs
        
        input_length = inputs["input_ids"].shape[-1]
        
        # max_length = generate_kwargs.get("max_length", 256)
        # if input_length < max_length:
        #     logger.warning(
        #         "Your max_length is set to {}, but you input_length is only {}. You might consider decreasing max_length manually, e.g. summarizer('...', max_length=50)".format(
        #             max_length, input_length
        #         )
        #     )

        outs = self.model.generate(
            input_ids=inputs['input_ids'].to(self.device), 
            attention_mask=inputs['attention_mask'].to(self.device),
            **generate_kwargs
        )

        prediction = self.tokenizer.decode(outs[0], skip_special_tokens=True)
        questions = prediction.split("<sep>")
        questions = [question.strip() for question in questions[:-1]]
        return questions
    
    def _prepare_inputs_for_e2e_qg(self, context):
        if self.language == LANGUAGE_PORTUGUES:
            source_text = f"gerar perguntas: {context}"
        else:
            source_text = f"generate questions: {context}"
        source_text = source_text + " </s>"
        
        inputs = self._tokenize([source_text], padding=False)
        return inputs
    
    def _tokenize(
        self,
        inputs,
        padding=True,
        truncation=True,
        add_special_tokens=True,
        max_length=512
    ):
        inputs = self.tokenizer.batch_encode_plus(
            inputs, 
            max_length=max_length,
            add_special_tokens=add_special_tokens,
            truncation=truncation,
            padding="max_length" if padding else False,
            pad_to_max_length=padding,
            return_tensors="pt"
        )
        return inputs
from typing import Optional, List
import torch

from transformers.models.blip.modeling_blip_text import BlipTextLMHeadModel, BlipTextModel
from transformers import BlipForConditionalGeneration, BlipConfig
from transformers.models.blip.modeling_blip_text import BlipTextOnlyMLMHead


class Blippa(BlipForConditionalGeneration):
    def __init__(self,config:BlipConfig,*args,**kwargs):
        super().__init__(config,*args,**kwargs)
        self.text_decoder = BlippaDecoder(self.text_decoder.config)

    
    


    def forward(
        self,
        pixel_values: torch.FloatTensor,
        input_ids: torch.FloatTensor = None,
        attention_mask: Optional[torch.LongTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        labels: Optional[torch.LongTensor] = None,
        return_dict: Optional[bool] = None,
        interpolate_pos_encoding: bool = False,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # step 1: forward the images through the vision encoder,
        # to get image embeddings of shape (batch_size, seq_len, hidden_size)
        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            interpolate_pos_encoding=interpolate_pos_encoding,
        )
        
        image_embeds = vision_outputs[0]
        batch_size = pixel_values.shape[0]
        if input_ids is None:
            input_ids = (
                torch.LongTensor([[self.config.text_config.bos_token_id]])
                .repeat(batch_size, 1)
                .to(image_embeds.device)
            )
        

        outputs = self.text_decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=image_embeds,
            labels=labels,
            return_dict=return_dict,
            reduction="mean",
            return_logits = True
        )

        return outputs,image_embeds
    





class BlippaDecoder(BlipTextLMHeadModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BlipTextModel(config, add_pooling_layer=False)
        self.cls = BlipTextOnlyMLMHead(config)
        self.label_smoothing = config.label_smoothing
        
    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings
        self.cls.predictions.bias = new_embeddings.bias

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.Tensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        return_logits: Optional[bool] = False,
        is_decoder: Optional[bool] = True,
        reduction: Optional[str] = "mean",
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        if labels is not None:
            use_cache = False

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            is_decoder=is_decoder,
        )
        sequence_output = outputs[0]
        
        return sequence_output
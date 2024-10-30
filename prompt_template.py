# prompt settings

class SentimentAnalysisTemplates:
    def __init__(self):
        self.system_prompt = "You are an expert of sentiment and opinion analysis."
        
    def set_system_prompt(self, prompt):
        self.system_prompt = prompt

    def prompt_direct_inferring(self, context, target):
        new_context = f'Given the sentence "{context}", '
        prompt = f"""{new_context} what is the sentiment polarity towards {target}? """
        return new_context, prompt

    def prompt_for_aspect(self, context, target):
        new_context = f'Given the sentence "{context}", '
        prompt = f"""{new_context} which specific aspect of {target} is possibly mentioned? 
        Answer briefly without explanation.
        """
        return new_context, prompt

    def prompt_for_opinion(self, context, target, aspect_expr):
        new_context = f"{context} and the mentioned aspect is about {aspect_expr}."
        prompt = f""" {new_context} 
        Based on the common sense, what is the implicit opinion towards the mentioned aspect of {target}, and why? 
        Answer briefly without explanation.
        """
        return new_context, prompt

    def prompt_for_polarity(self, context, target, opinion_expr):
        new_context = f"{context} The opinion towards the mentioned aspect of {target} is: {opinion_expr}."
        prompt = f""" {new_context} 
        Based on such opinion, what is the sentiment polarity towards {target}?
        Return only one of the sentiment polarity words: [positive, neutral, negative]
        """
        return new_context, prompt

    def prompt_for_polarity_label(self, context, polarity_expr):
        new_context = f"{context} The sentiment polarity is: {polarity_expr}."
        prompt = f""" {new_context} 
        Please label the sentiment polarity towards the target sentence.
        """
        return new_context, prompt
    
    def prompt_for_aspect(self, context, target):
        new_context = f'Given the sentence "{context}", '
        prompt = f"""{new_context} which specific aspect of {target} is possibly mentioned? 
        Answer briefly without explanation.
        """
        return new_context, prompt

    def prompt_for_opinion(self, context, target, aspect_expr):
        new_context = f"{context} and the mentioned aspect is about {aspect_expr}."
        prompt = f""" {new_context} 
        Based on the common sense, what is the implicit opinion towards the mentioned aspect of {target}, and why? 
        Answer briefly without explanation.
        """
        return new_context, prompt

    def prompt_for_polarity(self, context, target, opinion_expr):
        new_context = f"{context} The opinion towards the mentioned aspect of {target} is: {opinion_expr}."
        prompt = f""" {new_context} 
        Based on such opinion, what is the sentiment polarity towards {target}?
        Return only one of the sentiment polarity words: [positive, neutral, negative]
        """
        return new_context, prompt
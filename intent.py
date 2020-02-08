from nlp_engine import print_confidence_table
# pass through sub-intent model
def applyLeave(models, feature, context):
    print('Ok Applying for leave')
    subintent = models['typeofleave'].predict(feature)
    confidence_scores = models['typeofleave'].predict_proba(feature)
    confidence_scores = [round(score, 2) for score in confidence_scores[0]]
    print_confidence_table(models['typeofleave'].classes_, confidence_scores)

    if subintent == 'otherType':
        print('What type of leave do you want to apply?')


def smalltalk(models, feature, context):
    intent = models['smalltalk'].predict(feature)[0]
    confidence_scores = models['smalltalk'].predict_proba(feature)
    confidence_scores = [round(score, 2) for score in confidence_scores[0]]
    print_confidence_table(models['smalltalk'].classes_, confidence_scores)
    print(intent)
    responses = {
        'affirmative' : 'Alright! I\'ll do that', 
        'greet' : 'Hello!',
        'bye' : 'See you later!',
        'negative' : 'Cancelled!'
    }
    print('BOT: ' + responses[intent] + '\n')



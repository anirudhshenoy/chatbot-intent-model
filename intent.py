from nlp_engine import *
# pass through sub-intent model
# train main model with all leave data
def applyLeave(nlp, feature):
    nlp.push_context('applyLeave')
    typeOfLeave(nlp, feature)

    if nlp.check_last_context() == 'applyLeave':
        print('What type of leave do you want to apply?')
        feature = featurize(input(), nlp.vectors).reshape(1,-1)
        typeOfLeave(nlp, feature)

def typeOfLeave(nlp, feature):
    #nlp.push_context('typeofleave')
    subintent = run_model(nlp.models, 'typeofleave', feature)

    if subintent == 'maternity':
        nlp.push_context('maternity')
        print('Congratulations! Shall I apply for maternity leave on your behalf?')

    elif subintent == 'sick':
        nlp.push_context('sick')
        print('Hope you feel better! Shall I apply for sick leave on your behalf?')

    elif subintent == 'otherType':
        #nlp.push_context('otherType')
        pass
    elif subintent == 'smalltalk':
        smalltalk(nlp, feature)
        if nlp.check_last_context() == 'applyLeave':
            print('What type of leave do you want to apply?')
            feature = featurize(input(), nlp.vectors).reshape(1,-1)
            applyLeave(nlp, feature)




def smalltalk(nlp, feature):
    nlp.push_context('smalltalk')
    subintent = run_model(nlp.models, 'smalltalk', feature)[0]
    nlp.push_context(subintent)
    print(nlp.context)

    maternity_response = {
        'affirmative' : 'Alright I\'ve sent the request. Congrats on the baby!\n',
        'negative' : 'Canceling your request. Feel free to come back.\n',
    }

    sick_response = {
        'affirmative' : 'Alright I\'ve sent the request. Please take care!\n',
        'negative' : 'Canceling your sick leaverequest. Feel free to come back.\n',
    }

    applyLeave_smalltalk = {
        'affirmative' : 'You will have to tell me what type of leave you want to take!',
        'negative' : 'Alright! Not gonna apply for leave',
        'bye' : 'See ya!',
        'greet' : 'Hi There!'

    }

    main_smalltalk = {
        'bye' : 'See ya!',
        'greet' : 'Hi There!'
    }

    applyLeave_response =  {
            'maternity' : maternity_response,
            'sick' : sick_response,
            'otherType' : 'What type of leave would you like to apply for?',
            'smalltalk' : applyLeave_smalltalk
    }

    responses = {
        'applyLeave' : applyLeave_response,
        'leaveBalance' : 'Leave balance',
        'smalltalk' : main_smalltalk
    }

    temp_response = responses
    for context in nlp.context:
        temp_response = temp_response[context]
    if subintent == 'affirmative' or subintent == 'negative' or subintent == 'bye':
        nlp.clear_context()
    else:
        nlp.pop_context()
        nlp.pop_context()
    print(nlp.context)
    print(temp_response)

    """

   replies = {
        'leaveBalance' : 'Checking your balance!!',
        'bookMeeting' : 'Ok scheduling a meeting',
        'applyLeave' : {
            'maternity' : 'Congratulations! I will send your request for maternity leave',
            'sick' : 'Hope you feel better soon! Sending your request for Sick leave',
            'otherType' : 'What type of leave would you like to apply for?'
        },
        'smalltalk' : smalltalk
    }
    """
import pstep

print('Calling dispatch with List[int] and Dict[str, Any]')
pstep.dispatch([1, 2, 3], {
    'testing': "string",
    'integer': 5,
    });
print('Call successful.')

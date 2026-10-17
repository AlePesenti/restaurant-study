/**
 * Google Apps Script that creates the questionnaire for the study.
 *
 * 1. Go to https://script.google.com > New project, paste this file, save.
 * 2. Select the function createStudyForm and click Run (allow the permissions).
 * 3. Open the Execution log: it prints the form links and the PREFILLED LINK
 *    to put in webapp/study_config.json ("form_url").
 *
 * The same form is filled in once per assistant (four times per participant).
 * Section 1 holds the codes the web app fills in. "Assistant number" 1 leads to the
 * demographics section; 2-4 skip it. Every response ends with the assistant questions.
 */
function createStudyForm() {
  var form = FormApp.create('Restaurant assistants study')
    .setDescription('Questions about the assistant you just talked to. ' +
                    'There are no right or wrong answers.')
    .setProgressBar(true)
    .setAllowResponseEdits(false)
    .setLimitOneResponsePerUser(false)
    .setShowLinkToRespondAgain(false)
    .setConfirmationMessage("Thank you! Please go back to the study tab and click " +
                            "'I have submitted my answers, continue'.");
  try { form.setCollectEmail(false); } catch (e) {}
  try { form.setRequireLogin(false); } catch (e) {}   // only exists for Workspace accounts

  // ---- Section 1: codes (prefilled by the web app) ----
  var participant = form.addTextItem()
    .setTitle('Participant code')
    .setHelpText('Filled in automatically. Please do not change.')
    .setRequired(true);
  var assistant = form.addTextItem()
    .setTitle('Assistant code')
    .setHelpText('Filled in automatically. Please do not change.')
    .setRequired(true);
  var number = form.addMultipleChoiceItem()
    .setTitle('Assistant number')
    .setHelpText('Filled in automatically. Please do not change.')
    .setRequired(true);

  // ---- Section 2: about you (first assistant only) ----
  var aboutYou = form.addPageBreakItem().setTitle('About you');

  form.addMultipleChoiceItem().setTitle('Age')
    .setChoiceValues(['18-24', '25-34', '35-44', '45-54', '55-64', '65 or older'])
    .setRequired(true);
  form.addMultipleChoiceItem().setTitle('Gender')
    .setChoiceValues(['Female', 'Male', 'Non-binary', 'Prefer not to say'])
    .showOtherOption(true)
    .setRequired(true);
  form.addTextItem().setTitle('What is your native language?').setRequired(true);
  form.addMultipleChoiceItem().setTitle('How would you rate your English?')
    .setChoiceValues(['Native', 'Fluent', 'Good', 'Basic'])
    .setRequired(true);
  form.addMultipleChoiceItem()
    .setTitle('How often do you use voice assistants (e.g. Siri, Alexa, Google Assistant)?')
    .setChoiceValues(['Never', 'Less than once a month', 'Monthly', 'Weekly', 'Daily'])
    .setRequired(true);
  form.addMultipleChoiceItem().setTitle('How are you listening to the assistants?')
    .setChoiceValues(['Headphones or earbuds', 'Computer speakers', 'Phone speaker'])
    .showOtherOption(true)
    .setRequired(true);

  // ---- Section 3: about this assistant (every time) ----
  form.addPageBreakItem().setTitle('About this assistant')
    .setHelpText('Think about the assistant you just talked to.');

  scale(form, 'How natural did the voice sound?', 'Very unnatural', 'Very natural');
  scale(form, 'How easy was the voice to understand?', 'Very difficult', 'Very easy');
  scale(form, 'How pleasant was the voice to listen to?', 'Very unpleasant', 'Very pleasant');

  form.addSectionHeaderItem().setTitle('The assistant was...')
    .setHelpText('Please rate your impression of the assistant on these scales.');
  // Godspeed questionnaire (Bartneck et al., 2009): anthropomorphism, likeability, intelligence.
  scale(form, 'Fake - Natural', 'Fake', 'Natural');
  scale(form, 'Machinelike - Humanlike', 'Machinelike', 'Humanlike');
  scale(form, 'Artificial - Lifelike', 'Artificial', 'Lifelike');
  scale(form, 'Unfriendly - Friendly', 'Unfriendly', 'Friendly');
  scale(form, 'Unpleasant - Pleasant', 'Unpleasant', 'Pleasant');
  scale(form, 'Incompetent - Competent', 'Incompetent', 'Competent');
  scale(form, 'Unintelligent - Intelligent', 'Unintelligent', 'Intelligent');

  form.addMultipleChoiceItem().setTitle('Did you complete the task?')
    .setChoiceValues(['Yes', 'Partly', 'No'])
    .setRequired(true);
  scale(form, 'How satisfied are you with this assistant?', 'Very unsatisfied', 'Very satisfied');
  scale(form, "Would you trust this assistant's recommendation?", 'Not at all', 'Completely');
  form.addParagraphTextItem().setTitle('Any comments about this assistant? (optional)');

  // Branching: number 1 -> About you, 2-4 -> straight to the assistant questions.
  var assistantSection = form.getItems(FormApp.ItemType.PAGE_BREAK)[1].asPageBreakItem();
  number.setChoices([
    number.createChoice('1', aboutYou),
    number.createChoice('2', assistantSection),
    number.createChoice('3', assistantSection),
    number.createChoice('4', assistantSection)
  ]);

  try { form.setPublished(true); } catch (e) {}
  form.setAcceptingResponses(true);

  var prefilled = form.createResponse()
    .withItemResponse(participant.createResponse('PID'))
    .withItemResponse(assistant.createResponse('CODE'))
    .withItemResponse(number.createResponse('1'))
    .toPrefilledUrl();

  Logger.log('Edit the form:   ' + form.getEditUrl());
  Logger.log('Form link:       ' + form.getPublishedUrl());
  Logger.log('PREFILLED LINK (send this one): ' + prefilled);
}

function scale(form, title, low, high) {
  return form.addScaleItem().setTitle(title).setBounds(1, 5).setLabels(low, high).setRequired(true);
}

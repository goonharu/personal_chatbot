window.onload = function() {

// Variables
var messages = document.querySelector('.message-list')
var btn = document.querySelector('.btn')
var input = document.querySelector('input')
var recordBtn = document.querySelector('.record-btn')

var volume = 1;
var crossOrigin = 'anonymous';
let model;

let mediaRecorder;
let audioChunks = [];
let isRecording = false;

// Idle animation control
let idleInterval;
let blinkInterval;
let isSpeaking = false;

console.log('PIXI available:', typeof PIXI !== 'undefined');
console.log('PIXI.live2d available:', typeof PIXI !== 'undefined' && typeof PIXI.live2d !== 'undefined');

if (typeof PIXI === 'undefined') {
    console.error('❌ PIXI is not loaded! Check if the CDN scripts are loading.');
    return;
}

window.PIXI = PIXI;
const live2d = PIXI.live2d;

(async function () {
  let canvas_container = document.getElementById('canvas_container');
  const app = new PIXI.Application({
    view: document.getElementById('canvas'),
    autoStart: true,

    height: canvas_container.offsetHeight,
    width: canvas_container.offsetWidth,
    backgroundAlpha: 0.0,
    resizeTo: canvas_container
  });

  model = await live2d.Live2DModel.from('static/models/mafuyu/sub_mafuyumother_t05.model3.json', {
    autoInteract: false,
  });

  // Enhanced speak function with emotion, gesture, and sequential action support
  model.speakWithEmotion = async function (audioUrl, emotion = 'neutral', gesture = null, intensity = 0.5, actions = []) {
    isSpeaking = true;
    
    console.log('🎭 speakWithEmotion START:', {
        emotion,
        gesture,
        intensity,
        actions,
        actionsLength: actions ? actions.length : 0,
        audioUrl
    });
    
    // If we have actions, use them for sequential animation
    if (actions && actions.length > 0) {
      console.log(`🎬 Using sequential actions (${actions.length} actions)`);
      await playActionsSequentially(actions);
    } else {
      console.log(`🎭 Using single emotion/gesture (fallback)`);
      // Fallback to single emotion/gesture (legacy behavior)
      const faceMotion = getEmotionMotion(emotion, intensity);
      const bodyMotion = getGestureMotion(gesture);
      
      console.log(`📋 Calculated motions:`, {
          emotion,
          intensity,
          faceMotion,
          gesture,
          bodyMotion
      });
      
      if (faceMotion) {
        console.log(`👤 Triggering face motion: ${faceMotion}`);
        model.motion(faceMotion);
        await new Promise(resolve => setTimeout(resolve, 300));
      } else {
        console.warn(`⚠️ No face motion found for emotion: ${emotion}`);
      }
      
      if (bodyMotion) {
        console.log(`🚶 Triggering body motion: ${bodyMotion}`);
        model.motion(bodyMotion);
        await new Promise(resolve => setTimeout(resolve, 200));
      } else if (gesture) {
        console.warn(`⚠️ No body motion found for gesture: ${gesture}`);
      }
    }
    
    // Start audio with lip sync
    const audio = new Audio(audioUrl);
    const ctx = new AudioContext();
    const source = ctx.createMediaElementSource(audio);
    const analyser = ctx.createAnalyser();

    source.connect(analyser);
    analyser.connect(ctx.destination);

    const dataArray = new Uint8Array(analyser.frequencyBinCount);
    const coreModel = model.internalModel.coreModel;

    function updateMouth() {
      analyser.getByteTimeDomainData(dataArray);

      let sum = 0;
      for (let i = 0; i < dataArray.length; i++) {
        sum += Math.abs(dataArray[i] - 128);
      }

      const mouthOpen = Math.min(sum / dataArray.length / 30, 1);
      coreModel.setParameterValueById('ParamMouthOpenY', mouthOpen);

      if (!audio.paused) {
        requestAnimationFrame(updateMouth);
      }
    }

    audio.onplay = () => updateMouth();
    audio.onended = () => {
      coreModel.setParameterValueById('ParamMouthOpenY', 0);
      ctx.close();
      isSpeaking = false;
      
      // Return to neutral expression after speaking
      setTimeout(() => {
        model.motion('face_normal_01');
      }, 1000);
    };

    await audio.play();
  };

  app.stage.addChild(model);

  // Position model
  function positionModel() {
    const containerWidth = canvas_container.offsetWidth;
    const containerHeight = canvas_container.offsetHeight;
    
    const scaleToFit = (containerHeight * 0.9) / model.height;
    
    model.scale.set(scaleToFit, scaleToFit);
    model.anchor.set(0.5, 1.0);
    
    model.x = containerWidth / 2;
    model.y = containerHeight;
    
    // Initial pose
    model.motion('face_normal_01');

    console.log('Model positioned at:', model.x, model.y);
    console.log('Model scale:', scaleToFit);
  }

  positionModel();

  window.addEventListener('resize', () => {
    positionModel();
  });

  // Start idle animations after model is loaded
  startIdleAnimations();

})();

// Play actions sequentially with natural timing
async function playActionsSequentially(actions) {
  console.log(`🎬 Playing ${actions.length} actions sequentially...`);
  
  for (let i = 0; i < actions.length; i++) {
    const action = actions[i];
    console.log(`  Action ${i+1}/${actions.length}: ${action.text}`);
    
    // Trigger emotion if present
    if (action.emotion) {
      const faceMotion = getEmotionMotion(action.emotion, action.intensity);
      if (faceMotion) {
        console.log(`    → Face: ${faceMotion}`);
        model.motion(faceMotion);
        await new Promise(resolve => setTimeout(resolve, 400)); // Let expression settle
      }
    }
    
    // Trigger gesture if present
    if (action.gesture) {
      const bodyMotion = getGestureMotion(action.gesture);
      if (bodyMotion) {
        console.log(`    → Gesture: ${bodyMotion}`);
        model.motion(bodyMotion);
        await new Promise(resolve => setTimeout(resolve, 300)); // Let gesture play
      }
    }
    
    // Small pause between actions (if not last action)
    if (i < actions.length - 1) {
      await new Promise(resolve => setTimeout(resolve, 200));
    }
  }
  
  console.log(`✨ Finished playing all actions`);
}

// Emotion to motion mapping
function getEmotionMotion(emotion, intensity) {
  const emotionMap = {
    'happy': ['face_smile_01', 'face_smile_02', 'face_smile_03'],
    'excited': ['face_e_01', 'face_smile_04', 'face_smile_05'],
    'sad': ['face_sad_01', 'face_sad_02', 'face_sad_03', 'face_sad_04'],
    'angry': ['face_angry_01', 'face_angry_02', 'face_angry_03'],
    'surprised': ['face_surprise_01', 'face_surprise_02', 'face_surprise_03'],
    'worried': ['face_trouble_01', 'face_trouble_02', 'face_trouble_03l'],
    'serious': ['face_serious_01', 'face_serious_02'],
    'embarrassed': ['face_smallmouth_01'],
    'crying': ['face_cry_01', 'face_cry_02', 'face_cry_03', 'face_cry_04'],
    'neutral': ['face_normal_01']
  };
  
  const motions = emotionMap[emotion] || emotionMap['neutral'];
  
  // Use intensity to determine which variation (higher intensity = later variations)
  let index = Math.floor(intensity * motions.length);
  index = Math.min(index, motions.length - 1);
  
  return motions[index];
}

// Gesture to motion mapping
function getGestureMotion(gesture) {
  const gestureMap = {
    'nod': 's-common-nod01',
    'shake': 's-common-shakehead01',
    'think': 's-common-tilthead01',
    'greet': 's-common-forward01',
    'lookdown': 's-common-lookdown01'
  };
  
  return gestureMap[gesture] || null;
}

// Idle animation system
function startIdleAnimations() {
  console.log('Starting idle animations...');
  
  // Blinking animation (every 3-7 seconds)
  blinkInterval = setInterval(() => {
    if (!isSpeaking) {
      const blinkMotions = ['face_closeeye_01', 'face_closeeye_02', 'face_closeeye_03', 'face_closeeye_04'];
      const randomBlink = blinkMotions[Math.floor(Math.random() * blinkMotions.length)];
      model.motion(randomBlink);
    }
  }, Math.random() * 4000 + 3000); // 3-7 seconds
  
  // Random idle motions (every 10-20 seconds)
  idleInterval = setInterval(() => {
    if (!isSpeaking) {
      const idleMotions = [
        's-common-tilthead01',
        's-common-tilthead02',
        's-common-pose01',
        'face_normal_01'
      ];
      const randomIdle = idleMotions[Math.floor(Math.random() * idleMotions.length)];
      model.motion(randomIdle);
      console.log(`Idle motion: ${randomIdle}`);
    }
  }, Math.random() * 10000 + 10000); // 10-20 seconds
}

function messageInteraction(audio_link, emotion, gesture, intensity, actions) {
  console.log('🎬 messageInteraction called with:', {
      audio_link,
      emotion,
      gesture, 
      intensity,
      actions,
      actionsLength: actions ? actions.length : 0,
      modelExists: !!model,
      functionExists: model && typeof model.speakWithEmotion === 'function'
  });
  
  if (!model || typeof model.speakWithEmotion !== 'function') {
    console.warn('⚠️ Model not ready yet - waiting for initialization');
    return;
  }

  model.speakWithEmotion(audio_link, emotion, gesture, intensity, actions);
}

// Audio Recording function
async function toggleRecording() {
    if (!isRecording) {
        await startRecording();
    } else {
        stopRecording();
    }
}

async function startRecording() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });

        mediaRecorder = new MediaRecorder(stream);
        audioChunks = [];

        mediaRecorder.ondataavailable = (event) => {
            if (event.data.size > 0) {
                audioChunks.push(event.data);
            }
        };

        mediaRecorder.onstop = async () => {
            const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
            await sendAudioMessage(audioBlob);

            stream.getTracks().forEach(track => track.stop());
        };

        mediaRecorder.start();
        isRecording = true;

        recordBtn.innerHTML = '⏹️ Stop';
        recordBtn.classList.add('recording');

        console.log('Recording started');
    } catch (error) {
        console.error('Error accessing microphone:', error);
        alert('Could not access microphone. Please check permissions.');
    }
}

function stopRecording() {
    if (mediaRecorder && mediaRecorder.state !== 'inactive') {
        mediaRecorder.stop();
        isRecording = false;

        recordBtn.innerHTML = '🎤 Record';
        recordBtn.classList.remove('recording');
        
        console.log('Recording stopped');
    }
}

async function sendAudioMessage(audioBlob) {
    const formData = new FormData();
    formData.append('audio', audioBlob, 'recording.wav');

    try {
        writeLine(`<span>System</span><br> Transcribing audio...`, 'system')

        const response = await fetch('/transcribe', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error('Transcription failed');
        }

        const data = await response.json();

        if (data.error) {
            console.error('Transcription error:', data.error);
            writeLine(`<span>System</span><br> Error: ${data.error}`, 'system')
            return;
        }

        const lastMessage = messages.lastChild;
        if (lastMessage && lastMessage.innerHTML.includes('Transcribing')) {
            messages.removeChild(lastMessage);
        }

        input.value = data.transcription;
        sendMessage();
    } catch (error) {
        console.error('Error sending audio:', error);
        writeLine(`<span>System</span><br> Error processing audio. Please try again.`, 'system')
    }
}

// Button/Enter Key
btn.addEventListener('click', sendMessage)
recordBtn.addEventListener('click', toggleRecording)
input.addEventListener('keyup', function(e){ if(e.keyCode == 13) sendMessage() })

function loadHistory() {
    fetch('/history')
    .then(response => response.json())
    .then(data => {
        for (let i = 0; i < data.length; i++) {
            if (data[i].role === 'user') {
                writeLine(`<span>User</span><br> ${data[i].content}`, 'primary')
            } else if (data[i].role === 'assistant') {
                writeLine(`<span>Nari</span><br> ${data[i].content}`, 'secondary')
            }
        }
    })
}

loadHistory()

// Messenger Functions
function sendMessage(){
   var msg = input.value;
    if(msg.trim() == '') return;
   writeLine(`<span>User</span><br> ${msg}`, 'primary')
   input.value = ''
   
   fetch('/chat', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({'message': msg})
   })
   .then(response => response.json())
   .then(data => addMessage(data, 'secondary'))
   .catch(error => console.error('Error:', error));
}

function addMessage(msg, typeMessage='primary'){
   writeLine(`<span>${msg.FROM}</span><br> ${msg.MESSAGE}`, typeMessage)
   
   // Debug: Log what we received from backend
   console.log('📦 Received message data:', {
       FROM: msg.FROM,
       EMOTION: msg.EMOTION,
       GESTURE: msg.GESTURE,
       INTENSITY: msg.INTENSITY,
       ACTIONS: msg.ACTIONS,
       actionsLength: msg.ACTIONS ? msg.ACTIONS.length : 0,
       WAV: msg.WAV
   });
   
   // Pass emotion, gesture, intensity, and actions to animation
   messageInteraction(
       msg.WAV, 
       msg.EMOTION || 'neutral', 
       msg.GESTURE || null, 
       msg.INTENSITY || 0.5,
       msg.ACTIONS || []  // Sequential actions
   )
}

function writeLine(text, typeMessage){
   var message = document.createElement('li')
   message.classList.add('message-item', 'item-' + typeMessage)
   message.innerHTML = text
   messages.appendChild(message)
   messages.scrollTop = messages.scrollHeight;
}

}
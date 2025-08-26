const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const context = canvas.getContext('2d');

// Initialize SocketIO
const socket = io();

function startVideoStream() {
  navigator.mediaDevices.getUserMedia({ video: true })
    .then(stream => {
      video.srcObject = stream;
      video.onloadedmetadata = () => {
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        drawFrame();
      };
    })
    .catch(err => {
      console.error('Error accessing the camera: ', err);
    });
}

function drawFrame() {
  context.drawImage(video, 0, 0, canvas.width, canvas.height);
  requestAnimationFrame(drawFrame);
}

startVideoStream();

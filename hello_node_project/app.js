const express = require("express");
const path = require("path");
const fs = require("fs");
const { spawn } = require("child_process");

let stopSignal = false;
let pythonProcess = null; // Track the Python process

const app = express();
const port = 3000;

// Middleware to parse JSON data
app.use(express.json({ limit: "50mb" }));

// Serve the front-end HTML file
app.get("/", (req, res) => {
  res.sendFile(path.join(__dirname, "public", "index.html"));
});

// Trigger the Python script for gaze tracking
app.get("/trigger", (req, res) => {
  stopSignal = false; // Reset the stop signal before starting
  const scriptPath = path.join(__dirname, "Scripts", "gaze_tracking.py");

  // Run the Python script using spawn
  pythonProcess = spawn("python", [scriptPath]);

  pythonProcess.stdout.on("data", (data) => {
    console.log(`Python script output: ${data}`);
  });

  pythonProcess.stderr.on("data", (data) => {
    console.error(`Python script stderr: ${data}`);
  });

  pythonProcess.on("close", (code) => {
    console.log(`Python script exited with code ${code}`);
  });

  res.send("Gaze tracking script started!"); // Send response immediately
});

// Receive frames from the browser
app.post("/upload_frame", (req, res) => {
  const base64Data = req.body.image.replace(/^data:image\/jpeg;base64,/, "");

  
  // Define the path to store the image inside the Scripts folder
  const imagePath = path.join(__dirname, "Scripts", "current_frame.jpg");

  // Save the frame as an image file in the Scripts folder
  fs.writeFile(imagePath, base64Data, "base64", (err) => {
    if (err) {
      console.log("Error saving image:", err);
      return res.status(500).send("Error saving image");
    }
  });

  // Respond after the frame is received and saved
  res.send("Frame received");
});

// Route to stop the Python script
app.get("/stop", (req, res) => {
  stopSignal = true;
  res.send("Gaze tracking stopped!");
});

// Route to check stop signal
app.get("/stop_signal", (req, res) => {
  if (stopSignal) {
    res.send("STOP");
  } else {
    res.send("RUNNING");
  }
});

// Run the server
app.listen(port, () => {
  console.log(`Server running at http://localhost:${port}`);
});

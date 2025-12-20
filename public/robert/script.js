import { initializeApp } from "https://www.gstatic.com/firebasejs/12.7.0/firebase-app.js";
import { getFirestore, doc, getDoc, setDoc, onSnapshot } from "https://www.gstatic.com/firebasejs/12.7.0/firebase-firestore.js";
import { getAuth, signInAnonymously, onAuthStateChanged } from "https://www.gstatic.com/firebasejs/12.7.0/firebase-auth.js";

// ----- Firebase Config -----
const firebaseConfig = {
    apiKey: "AIzaSyB3d0SHTnCh1TCqQ_p7EK5IWaLaMjacAro",
    authDomain: "robertgotchi.firebaseapp.com",
    projectId: "robertgotchi",
    storageBucket: "robertgotchi.firebasestorage.app",
    messagingSenderId: "439033305646",
    appId: "1:439033305646:web:783dae55db72f2d8339dbf"
};

const app = initializeApp(firebaseConfig);
const db = getFirestore(app);
const auth = getAuth();

let petDoc;
const pet = {
    hunger: 40,
    energy: 60,
    health: 80,
    alive: true,
    lastUpdate: Date.now(),
    lastAction: 0,
};

// ----- Authentication -----
signInAnonymously(auth);

onAuthStateChanged(auth, (user) => {
    if (user) {
        const uid = user.uid;
        console.log("User ID:", uid);

        // Each user gets their own Tamagotchi
        petDoc = doc(db, "pets", uid);
        initPet();
    }
});

// ----- Initialize Pet -----
async function initPet() {
    const snap = await getDoc(petDoc);

    if (snap.exists()) {
        const data = snap.data();
        pet.hunger = data.hunger;
        pet.energy = data.energy;
        pet.health = data.health;
        pet.alive = data.alive;
        pet.lastUpdate = data.lastUpdated;
    } else {
        await setDoc(petDoc, {
            hunger: pet.hunger,
            energy: pet.energy,
            health: pet.health,
            alive: pet.alive,
            lastUpdated: Date.now(),
            lastAction: 0,
        });
    }

    // Listen for updates (so UI reacts if something changes)
    onSnapshot(petDoc, (snap) => {
        if (snap.exists()) {
            const data = snap.data();
            pet.hunger = data.hunger;
            pet.energy = data.energy;
            pet.health = data.health;
            pet.alive = data.alive;
            pet.lastUpdate = data.lastUpdated;

            applyDecay();
            updateUI();
        }
    });

    // Start local auto-decay loop
    setInterval(() => {
        applyDecay();
        updateUI();
    }, 1000);
}

// ----- Decay -----
function applyDecay() {
    const now = Date.now();
    const minutesPassed = (now - pet.lastUpdate) / 60000;
    pet.lastUpdate = now;

    pet.hunger += 5 * minutesPassed;
    pet.energy -= 2.5 * minutesPassed;
    pet.health -= 1.0 * minutesPassed;

    if (pet.hunger >= 100 || pet.energy <= 0 || pet.health <= 0) {
        pet.alive = false;
    } else {
        pet.alive = true;
    }

    clamp();
}

// ----- Clamp stats -----
function clamp() {
    pet.hunger = Math.max(0, Math.min(100, pet.hunger));
    pet.energy = Math.max(0, Math.min(100, pet.energy));
    pet.health = Math.max(0, Math.min(100, pet.health));
}

// ----- Update UI -----
function getEmotion() {
    if (!pet.alive) return "dead";
    if (pet.health < 20) return "weak";
    if (pet.health < 40) return "sick";
    if (pet.energy < 20) return "sleepy";
    if (pet.hunger > 80) return "angry";
    if (pet.hunger > 60) return "sad";
    if (pet.hunger < 30 && pet.health > 70) return "happy";
    return "content";
}

function updateUI() {
    const emotion = getEmotion();
    document.getElementById("statusText").textContent = `Feeling ${emotion}`;
    document.getElementById("petImage").src = `images/${emotion}.png`;
    document.getElementById("hungerBar").style.width = pet.hunger + "%";
    document.getElementById("energyBar").style.width = pet.energy + "%";
    document.getElementById("healthBar").style.width = pet.health + "%";
}

// ----- Rate-limited Firestore update -----
async function updatePetInFirestore() {
    const now = Date.now();

    await setDoc(petDoc, {
        hunger: pet.hunger,
        energy: pet.energy,
        health: pet.health,
        alive: pet.alive,
        lastUpdated: now,
        lastAction: now,
    }, { merge: true });
}

// ----- Actions -----

async function canAct() {
    const snap = await getDoc(petDoc);
    const data = snap.data();

    const lastAction = data.lastAction || 0;
    const now = Date.now();
    const COOLDOWN = 5000; // 5 seconds

    if (now - lastAction < COOLDOWN) {
        const timeRemaining = COOLDOWN - (now - lastAction);
        const seconds = Math.ceil(timeRemaining / 1000);
        alert(`Please wait ${seconds} seconds before doing something with Robert.`);
        return false;
    }
    return true;
}

async function feed() {
    if (!(await canAct())) return;

    pet.hunger -= 20;
    pet.energy += 5;
    clamp();
    updateUI();
    await updatePetInFirestore();
}

async function sleep() {
    if (!(await canAct())) return;

    pet.energy += 30;
    pet.hunger += 10;
    clamp();
    updateUI();
    await updatePetInFirestore();
}

async function medicine() {
    if (!(await canAct())) return;

    pet.health += 25;
    pet.energy -= 5;
    clamp();
    updateUI();
    await updatePetInFirestore();
}

async function play() {
    if (!(await canAct())) return;

    pet.energy -= 15;
    pet.hunger += 10;
    pet.health += 5;
    clamp();
    updateUI();
    await updatePetInFirestore();
}

// ----- Button listeners -----
document.getElementById("feedBtn").addEventListener("click", feed);
document.getElementById("sleepBtn").addEventListener("click", sleep);
document.getElementById("medicineBtn").addEventListener("click", medicine);
document.getElementById("playBtn").addEventListener("click", play);

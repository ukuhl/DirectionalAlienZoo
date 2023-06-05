import StartScene from './StartScene.js';

class GetProlificID extends Phaser.Scene {

	constructor(varObj) {
		super({key : 'getProlificID'});
		this.varObj = varObj;
		this.startTime = undefined;
	}

	init() {}

	preload() {

        // load form to insert Prolific ID
        this.load.html("formProlific", "static/formProlificID.html");
        
        // load images
        this.load.image('UBIE', 'static/UBF-logo2.png');
		this.load.image('NINJA', 'static/NINJA_logo.png');
		this.load.image('buttonFeed', 'static/buttonSubmit.png');

	}

	create() {

        this.startTime = new Date().getTime();

		this.add.image(window.innerWidth * 0.1, window.innerHeight * 0.1, 'UBIE');
		this.add.image(window.innerWidth * 0.8, window.innerHeight * 0.1, 'NINJA');

        this.add.text(window.innerWidth * 0.05, window.innerHeight * 0.20, "Please provide your Prolific ID, and confirm by hitting the return key:", { fontFamily: 'Arial', fontSize: '17px', color: '#000000', align: 'left'});

        this.IDinput = this.add.dom(window.innerWidth * 0.3, window.innerHeight * 0.30).createFromCache("formProlific");

        this.returnKey = this.input.keyboard.addKey(Phaser.Input.Keyboard.KeyCodes.ENTER);

        var prolificID = ""

        this.returnKey.on("down", event => {
            let name = this.IDinput.getChildByName("name");
            this.varObj.prolificID = name.value;
            if(name.value != "") {
                this.add.text(window.innerWidth * 0.05, window.innerHeight * 0.40, "You have entered: "+name.value, { fontFamily: 'Arial', fontSize: '17px', color: '#000000', align: 'left'});
                this.add.text(window.innerWidth * 0.05, window.innerHeight * 0.45, "Please click the button below to continue.", { fontFamily: 'Arial', fontSize: '17px', color: '#000000', align: 'left'});
                name.value = "";

                // instatiate and add new start scene with current data
                var startScene = undefined;
        
                // add button to start game and switch to fullscreen
                var buttonStart = this.add.image(0, 0, 'buttonFeed').setScale(0.4)
                    .setInteractive()
                    .on('pointerdown', () => this.logTime())
                    .on('pointerdown', () => this.varObj.api.logUserPayment(this.varObj.prolificID))	// Store (encrypted) prolificID on server
                    .on('pointerdown', () => startScene = new StartScene(this.varObj))
                    .on('pointerdown', () => this.scene.add('startScene', startScene))
                    .on('pointerdown', () => this.scene.start('startScene'));
        
                var textStart = this.add.text(-50, -15, 'Continue!', { fontSize: '20px', color: '#ffffff' }).setOrigin(0);
                this.add.container(window.innerWidth * 0.85, window.innerHeight * 0.75, [buttonStart, textStart]);
            }
        });
	}

	logTime() {
		var time = new Date().getTime() - this.startTime;
		this.varObj.api.logTime(2, time);
	}

	update() {}

}

export default GetProlificID;

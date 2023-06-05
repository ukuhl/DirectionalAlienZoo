class EndScene extends Phaser.Scene {

	constructor(varObj) {
		super({key : 'endScene'});
		this.varObj = varObj
	}

	init() { }

	preload() {
		// load button images
		this.load.spritesheet('checkbox', 'static/CheckBoxSprites.png', { frameWidth: 51, frameHeight: 50 });
		this.load.image('UBIE', 'static/UBF-logo.png');
		this.load.image('NINJA', 'static/NINJA_logo.png');
		this.load.image('buttonAgree', 'static/buttonSubmit.png');

		// load form to display payment code
		this.load.html("formCompletion", "static/formCompletionCode.html");

	}

	create() {

		this.add.image(window.innerWidth * 0.1, window.innerHeight * 0.1, 'UBIE');
		this.add.image(window.innerWidth * 0.8, window.innerHeight * 0.1, 'NINJA');

		var qIntro1 = [
			'You have completed the study. Thank you very much for your participation!',
			'',
			'The study completion code is:',
		];

		var qIntro2 = [
			'Please copy down this code, return to Prolific and insert the code to validate that you completed the study.',
		];

		var qIntro3 = [
			'If you would like to know more about the aim of this work, click the button below.',
			'',
			'In case you have further questions or comments regarding this study, please contact',
			'',
			'Dr. Ulrike Kuhl at Bielefeld University: ukuhl@techfak.uni-bielefeld.de .',
			'',
			'You may close this window, after you have copied to code.',
			''
		];

		// add button to start game and switch to fullscreen
		var buttonDebrief = this.add.image(0, 0, 'buttonAgree').setScale(0.65)
			.setInteractive()
			.on('pointerdown', () => { window.open('https://its-ml.de/index.php/directional-alien-zoo-debriefing/'); });

		var textDebrief = this.add.text(-100, -25, ['I want to learn more','about the study!'], { fontSize: '18px', color: '#ffffff' }).setOrigin(0);
		var buttonContainer = this.add.container(window.innerWidth * 0.85, window.innerHeight * 0.80, [buttonDebrief, textDebrief])

		this.add.text(window.innerWidth * 0.05, window.innerHeight * 0.20, qIntro1, { fontFamily: 'Arial', fontSize: '18px', color: '#000000' });
		this.add.dom(window.innerWidth * 0.3, window.innerHeight * 0.30).createFromCache("formCompletion");
		this.add.text(window.innerWidth * 0.05, window.innerHeight * 0.40, qIntro2, { fontFamily: 'Arial', fontSize: '18px', fontStyle: "bold",  color: '#000000' });
		this.add.text(window.innerWidth * 0.05, window.innerHeight * 0.50, qIntro3, { fontFamily: 'Arial', fontSize: '18px', color: '#000000' });


	  }

	update() { }

}

export default EndScene;

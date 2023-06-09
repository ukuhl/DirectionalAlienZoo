import EndScene from './endScene.js';

class QuestionnaireScene4 extends Phaser.Scene {

	constructor(varObj) {
		super({ key: 'questionnaireScene5' });
		this.varObj = varObj;

		this.itemGenderVar1Checked = false;
		this.itemGenderVar2Checked = false;
		this.itemGenderVar3Checked = false;
		this.itemGenderVar4Checked = false;

		this.itemAgeVar1Checked = false;
		this.itemAgeVar2Checked = false;
		this.itemAgeVar3Checked = false;
		this.itemAgeVar4Checked = false;
		this.itemAgeVar5Checked = false;
		this.itemAgeVar6Checked = false;
		this.itemAgeVar7Checked = false;

	}

	init() { }

	preload() {
		// load button images
		this.load.spritesheet('checkbox', 'static/CheckBoxSprites.png', { frameWidth: 51, frameHeight: 50 });
	}

	create() {
		// clean slate:
		this.children.removeAll();

		var demographics = [
			'To finish up, we would like to collect some demographic information.',,
		]

		this.add.text(window.innerWidth * 0.025, window.innerHeight * 0.125, demographics, { fontFamily: 'Arial', fontSize: '17px', fontStyle: "bold", color: '#000000' });

		var itemGender = [
			"Which term most accurately describes your gender?"
		]

		// add item 1
		this.add.text(window.innerWidth * 0.025, window.innerHeight * 0.175, itemGender, { fontFamily: 'Arial', fontSize: '17px', fontStyle: "bold", color: '#000000' });

		// option 1
		this.add.text(window.innerWidth * 0.025, window.innerHeight * 0.225, 'Female', { fontFamily: 'Arial', fontSize: '17px', color: '#000000' });

		var itemGenderSprite1 = this.add.sprite(window.innerWidth * 0.04, window.innerHeight * 0.275, 'checkbox', 0).setScale(0.4).setInteractive();

		itemGenderSprite1.on('pointerdown', function (pointer) {
			if (this.itemGenderVar1Checked) {
				itemGenderSprite1.setFrame(0);
				this.itemGenderVar1Checked = false;
			} else {
				itemGenderSprite1.setFrame(1);
				this.itemGenderVar1Checked = true;

				this.itemGenderVar2Checked = false;
				this.itemGenderVar3Checked = false;
				this.itemGenderVar4Checked = false;

			}
		}.bind(this));

		// option 2
		this.add.text(window.innerWidth * 0.150, window.innerHeight * 0.225, 'Male', { fontFamily: 'Arial', fontSize: '17px', color: '#000000' });

		var itemGenderSprite2 = this.add.sprite(window.innerWidth * 0.16, window.innerHeight * 0.275, 'checkbox', 0).setScale(0.4).setInteractive();

		itemGenderSprite2.on('pointerdown', function (pointer) {
			if (this.itemGenderVar2Checked) {
				itemGenderSprite2.setFrame(0);
				this.itemGenderVar2Checked = false;
			} else {
				itemGenderSprite2.setFrame(1);
				this.itemGenderVar2Checked = true;

				this.itemGenderVar1Checked = false;
				this.itemGenderVar3Checked = false;
				this.itemGenderVar4Checked = false;

			}
		}.bind(this));

		// option 3
		this.add.text(window.innerWidth * 0.265, window.innerHeight * 0.21, 'Non-binary /\n gender non-\nconforming', { fontFamily: 'Arial', fontSize: '17px', color: '#000000' });

		var itemGenderSprite3 = this.add.sprite(window.innerWidth * 0.29, window.innerHeight * 0.275, 'checkbox', 0).setScale(0.4).setInteractive();

		itemGenderSprite3.on('pointerdown', function (pointer) {
			if (this.itemGenderVar3Checked) {
				itemGenderSprite3.setFrame(0);
				this.itemGenderVar3Checked = false;
			} else {
				itemGenderSprite3.setFrame(1);
				this.itemGenderVar3Checked = true;

				this.itemGenderVar1Checked = false;
				this.itemGenderVar2Checked = false;
				this.itemGenderVar4Checked = false;

			}
		}.bind(this));

		// option 4
		this.add.text(window.innerWidth * 0.385, window.innerHeight * 0.21, 'I prefer not to answer.', { fontFamily: 'Arial', fontSize: '17px', color: '#000000' });

		var itemGenderSprite4 = this.add.sprite(window.innerWidth * 0.42, window.innerHeight * 0.275, 'checkbox', 0).setScale(0.4).setInteractive();

		itemGenderSprite4.on('pointerdown', function (pointer) {
			if (this.itemGenderVar4Checked) {
				itemGenderSprite4.setFrame(0);
				this.itemGenderVar4Checked = false;
			} else {
				itemGenderSprite4.setFrame(1);
				this.itemGenderVar4Checked = true;

				this.itemGenderVar1Checked = false;
				this.itemGenderVar2Checked = false;
				this.itemGenderVar3Checked = false;

			}
		}.bind(this));

		var itemAge = [
			"Plase indicate your age:"
		]

		// add item 1
		this.add.text(window.innerWidth * 0.025, window.innerHeight * 0.35, itemAge, { fontFamily: 'Arial', fontSize: '17px', fontStyle: "bold", color: '#000000' });

		// option 1
		this.add.text(window.innerWidth * 0.025, window.innerHeight * 0.4, '18-24y', { fontFamily: 'Arial', fontSize: '17px', color: '#000000' });

		var itemAgeSprite1 = this.add.sprite(window.innerWidth * 0.04, window.innerHeight * 0.45, 'checkbox', 0).setScale(0.4).setInteractive();

		itemAgeSprite1.on('pointerdown', function (pointer) {
			if (this.itemAgeVar1Checked) {
				itemAgeSprite1.setFrame(0);
				this.itemAgeVar1Checked = false;
			} else {
				itemAgeSprite1.setFrame(1);
				this.itemAgeVar1Checked = true;

				this.itemAgeVar2Checked = false;
				this.itemAgeVar3Checked = false;
				this.itemAgeVar4Checked = false;
				this.itemAgeVar5Checked = false;
				this.itemAgeVar6Checked = false;
				this.itemAgeVar7Checked = false;
				itemAgeSprite2.setFrame(0);
				itemAgeSprite3.setFrame(0);
				itemAgeSprite4.setFrame(0);
				itemAgeSprite5.setFrame(0);
				itemAgeSprite6.setFrame(0);

			}
		}.bind(this));

		// option 2
		this.add.text(window.innerWidth * 0.125, window.innerHeight * 0.4, '25-34y', { fontFamily: 'Arial', fontSize: '17px', color: '#000000' });

		var itemAgeSprite2 = this.add.sprite(window.innerWidth * 0.14, window.innerHeight * 0.45, 'checkbox', 0).setScale(0.4).setInteractive();

		itemAgeSprite2.on('pointerdown', function (pointer) {
			if (this.itemAgeVar2Checked) {
				itemAgeSprite2.setFrame(0);
				this.itemAgeVar2Checked = false;
			} else {
				itemAgeSprite2.setFrame(1);
				this.itemAgeVar2Checked = true;

				this.itemAgeVar1Checked = false;
				this.itemAgeVar3Checked = false;
				this.itemAgeVar4Checked = false;
				this.itemAgeVar5Checked = false;
				this.itemAgeVar6Checked = false;
				this.itemAgeVar7Checked = false;
				itemAgeSprite1.setFrame(0);
				itemAgeSprite3.setFrame(0);
				itemAgeSprite4.setFrame(0);
				itemAgeSprite5.setFrame(0);
				itemAgeSprite6.setFrame(0);

			}
		}.bind(this));

		// option 3
		this.add.text(window.innerWidth * 0.225, window.innerHeight * 0.4, '35-44y', { fontFamily: 'Arial', fontSize: '17px', color: '#000000' });

		var itemAgeSprite3 = this.add.sprite(window.innerWidth * 0.24, window.innerHeight * 0.45, 'checkbox', 0).setScale(0.4).setInteractive();

		itemAgeSprite3.on('pointerdown', function (pointer) {
			if (this.itemAgeVar3Checked) {
				itemAgeSprite3.setFrame(0);
				this.itemAgeVar3Checked = false;
			} else {
				itemAgeSprite3.setFrame(1);
				this.itemAgeVar3Checked = true;

				this.itemAgeVar1Checked = false;
				this.itemAgeVar2Checked = false;
				this.itemAgeVar4Checked = false;
				this.itemAgeVar5Checked = false;
				this.itemAgeVar6Checked = false;
				this.itemAgeVar7Checked = false;
				itemAgeSprite1.setFrame(0);
				itemAgeSprite2.setFrame(0);
				itemAgeSprite4.setFrame(0);
				itemAgeSprite5.setFrame(0);
				itemAgeSprite6.setFrame(0);

			}
		}.bind(this));

		// option 4
		this.add.text(window.innerWidth * 0.325, window.innerHeight * 0.4, '45-54y', { fontFamily: 'Arial', fontSize: '17px', color: '#000000' });

		var itemAgeSprite4 = this.add.sprite(window.innerWidth * 0.34, window.innerHeight * 0.45, 'checkbox', 0).setScale(0.4).setInteractive();

		itemAgeSprite4.on('pointerdown', function (pointer) {
			if (this.itemAgeVar4Checked) {
				itemAgeSprite4.setFrame(0);
				this.itemAgeVar4Checked = false;
			} else {
				itemAgeSprite4.setFrame(1);
				this.itemAgeVar4Checked = true;

				this.itemAgeVar1Checked = false;
				this.itemAgeVar2Checked = false;
				this.itemAgeVar3Checked = false;
				this.itemAgeVar5Checked = false;
				this.itemAgeVar6Checked = false;
				this.itemAgeVar6Checked = false;
				this.itemAgeVar7Checked = false;
				itemAgeSprite1.setFrame(0);
				itemAgeSprite2.setFrame(0);
				itemAgeSprite3.setFrame(0);
				itemAgeSprite5.setFrame(0);
				itemAgeSprite6.setFrame(0);

			}
		}.bind(this));

		// option 5
		this.add.text(window.innerWidth * 0.425, window.innerHeight * 0.4, '55-64y', { fontFamily: 'Arial', fontSize: '17px', color: '#000000' });

		var itemAgeSprite5 = this.add.sprite(window.innerWidth * 0.44, window.innerHeight * 0.45, 'checkbox', 0).setScale(0.4).setInteractive();

		itemAgeSprite5.on('pointerdown', function (pointer) {
			if (this.itemAgeVar5Checked) {
				itemAgeSprite5.setFrame(0);
				this.itemAgeVar5Checked = false;
			} else {
				itemAgeSprite5.setFrame(1);
				this.itemAgeVar5Checked = true;

				this.itemAgeVar1Checked = false;
				this.itemAgeVar2Checked = false;
				this.itemAgeVar3Checked = false;
				this.itemAgeVar4Checked = false;
				this.itemAgeVar6Checked = false;
				this.itemAgeVar7Checked = false;
				itemAgeSprite1.setFrame(0);
				itemAgeSprite2.setFrame(0);
				itemAgeSprite3.setFrame(0);
				itemAgeSprite4.setFrame(0);
				itemAgeSprite6.setFrame(0);

			}
		}.bind(this));

		// option 6
		this.add.text(window.innerWidth * 0.525, window.innerHeight * 0.4, '65y and over', { fontFamily: 'Arial', fontSize: '17px', color: '#000000' });

		var itemAgeSprite6 = this.add.sprite(window.innerWidth * 0.54, window.innerHeight * 0.45, 'checkbox', 0).setScale(0.4).setInteractive();

		itemAgeSprite6.on('pointerdown', function (pointer) {
			if (this.itemAgeVar6Checked) {
				itemAgeSprite6.setFrame(0);
				this.itemAgeVar6Checked = false;
			} else {
				itemAgeSprite6.setFrame(1);
				this.itemAgeVar6Checked = true;

				this.itemAgeVar1Checked = false;
				this.itemAgeVar2Checked = false;
				this.itemAgeVar3Checked = false;
				this.itemAgeVar4Checked = false;
				this.itemAgeVar5Checked = false;
				this.itemAgeVar7Checked = false;
				itemAgeSprite1.setFrame(0);
				itemAgeSprite2.setFrame(0);
				itemAgeSprite3.setFrame(0);
				itemAgeSprite4.setFrame(0);
				itemAgeSprite5.setFrame(0);
				itemAgeSprite7.setFrame(0);

			}
		}.bind(this));

		// Don't know
		this.add.text(window.innerWidth * 0.625, window.innerHeight * 0.4, 'I prefer not to answer.', { fontFamily: 'Arial', fontSize: '17px', color: '#000000' });

		var itemAgeSprite7 = this.add.sprite(window.innerWidth * 0.64, window.innerHeight * 0.45, 'checkbox', 0).setScale(0.4).setInteractive();

		itemAgeSprite7.on('pointerdown', function (pointer) {
			if (this.itemAgeVar7Checked) {
				itemAgeSprite7.setFrame(0);
				this.itemAgeVar7Checked = false;
			} else {
				itemAgeSprite7.setFrame(1);
				this.itemAgeVar7Checked = true;

				this.itemAgeVar1Checked = false;
				this.itemAgeVar2Checked = false;
				this.itemAgeVar3Checked = false;
				this.itemAgeVar4Checked = false;
				this.itemAgeVar5Checked = false;
				this.itemAgeVar6Checked = false;
				itemAgeSprite1.setFrame(0);
				itemAgeSprite2.setFrame(0);
				itemAgeSprite3.setFrame(0);
				itemAgeSprite4.setFrame(0);
				itemAgeSprite5.setFrame(0);
				itemAgeSprite6.setFrame(0);

			}
		}.bind(this));


		// instatiate and add new end scene with current data
		var endScene = undefined;

		// add button to request feedback
		var buttonContinue = this.add.image(0, 0, 'buttonFeed').setScale(0.4)
			.setInteractive()
			.on('pointerdown', () => this.onBtnContinue());

		var textContinue = this.add.text(-50, -15, 'Continue!', { fontSize: '18px', color: '#ffffff' })
		var buttonContainer = this.add.container(window.innerWidth * 0.85, window.innerHeight * 0.80, [buttonContinue, textContinue])

	}

	logAnswers() {
		this.varObj.api.logDemographics(this.itemAgeVar1Checked, this.itemAgeVar2Checked, this.itemAgeVar3Checked, this.itemAgeVar4Checked, this.itemAgeVar5Checked, this.itemAgeVar6Checked, this.itemAgeVar7Checked,
			this.itemGenderVar1Checked, this.itemGenderVar2Checked, this.itemGenderVar3Checked, this.itemGenderVar4Checked);
	}

	onBtnContinue() {
		if ([this.itemAgeVar1Checked, this.itemAgeVar2Checked, this.itemAgeVar3Checked, this.itemAgeVar4Checked, this.itemAgeVar5Checked, this.itemAgeVar6Checked, this.itemAgeVar7Checked].every(a => a == false) ||
		[this.itemGenderVar1Checked, this.itemGenderVar2Checked, this.itemGenderVar3Checked, this.itemGenderVar4Checked].every(a => a == false)) {
		

			// alerts cause issues in fullscreen mode, here's a workaround:

			// background
			var warnDialogBG = this.add.rectangle(0, 0, window.innerWidth, window.innerHeight, 0xFFFFFF, 0.5).setOrigin(0);
			// dialog
			var warnDialog = this.add.rectangle((window.innerWidth * 0.5)-200, (window.innerHeight * 0.5)-130, 400, 200, 0xFFFFFF, 1).setOrigin(0);
			warnDialog.setStrokeStyle(1, 0x1000000, 1);
			var warnDialogTxt = this.add.text((window.innerWidth * 0.5)-150, (window.innerHeight * 0.5)-80, "Please answer all questions!", { fontFamily: "Arial", fontSize: '25px', color: '#000000' }).setOrigin(0);
			// text
			var buttonOK = this.add.rectangle((window.innerWidth * 0.5)-50, (window.innerHeight * 0.5)-25, 100, 50, 0x1a65ac, 1).setOrigin(0);
			buttonOK.setStrokeStyle(1, 0x1000000, 1);
			var buttontextOK = this.add.text((window.innerWidth * 0.5)-20, (window.innerHeight * 0.5)-12.5, 'Ok', { fontFamily: "Arial", fontSize: '25px', color: '#FFFFFF' }).setOrigin(0);

			// make interactive and remove all after clicking ok
			buttonOK
				.setInteractive()
				.on('pointerdown', () => { warnDialogBG.destroy(); })
				.on('pointerdown', () => { warnDialog.destroy(); })
				.on('pointerdown', () => { warnDialogTxt.destroy(); })
				.on('pointerdown', () => { buttonOK.destroy(); })
				.on('pointerdown', () => { buttontextOK.destroy(); });

		}
		else {
			this.logAnswers();

			var endScene = new EndScene(this.varObj);
			this.scene.remove('endScene', endScene);
			this.scene.add('endScene', endScene);
			this.scene.start('endScene');
			this.scale.stopFullscreen();
		}
	}

	update() { }

}

export default QuestionnaireScene4;

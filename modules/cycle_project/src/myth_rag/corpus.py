"""
corpus.py — MYTH_RAG: Bundled flood/catastrophe myth corpus

All texts are accurate scholarly paraphrases / key passages from public domain
ancient sources. Temporal estimates (estimated_bp_*) represent hypothesized ages
of the events described — NOT composition dates. These estimates are speculative
ranges used for geological correlation, clearly labeled as such.

Sources downloadable by the user (real URLs in sources.py):
  Project Gutenberg: gutenberg.org
  Sacred Texts Archive: sacred-texts.com
  Internet Archive: archive.org
  ETCSL (Sumerian): etcsl.orinst.ox.ac.uk

Each entry:
  id              — unique identifier
  culture         — civilization/tradition
  region          — geographic origin
  title           — source text name
  text            — key catastrophe passage (scholarly paraphrase, public domain)
  themes          — list of semantic tags
  composition_bce — when the text was written/compiled (CE dates are negative)
  estimated_bp_min/max — hypothesized age of described events (years BP, speculative)
  temporal_notes  — reasoning for date estimate
  source_url      — real URL for user to download full text
"""

MYTHS = [

# ─────────────────────────────────────────────────────────────────────────────
# NEAR EAST
# ─────────────────────────────────────────────────────────────────────────────
{
    "id": "gilgamesh_flood",
    "culture": "Sumerian/Akkadian",
    "region": "Mesopotamia",
    "title": "Epic of Gilgamesh — Tablet XI: The Flood",
    "composition_bce": 2100,
    "estimated_bp_min": 10_000,
    "estimated_bp_max": 15_000,
    "temporal_notes": "Event placed after the antediluvian kings; some scholars associate with Black Sea infilling ~7500 BP or Younger Dryas transition ~12900 BP. Range reflects scholarly debate.",
    "source_url": "https://www.gutenberg.org/ebooks/11000",
    "themes": ["flood", "divine_warning", "boat_ark", "survival", "animals_saved",
                "rainbow", "seven_days_rain", "mountain_refuge", "dove_raven"],
    "text": """
Utnapishtim spoke to Gilgamesh: The secret of the gods I will reveal to thee.
The great gods resolved to send the Flood. Their father Anu swore by heaven.
Enlil, the counselor, swore. Ninurta the helper swore. Ennugi the canal-master swore.
Their words came to Ea, who repeated them to the reed fence:
'Reed house, reed house! Wall, wall! Hear, O reed house! Understand, O wall!
Man of Shuruppak, son of Ubartutu, tear down thy house, build a ship.
Abandon possessions, seek life. Forsake goods and keep the soul alive.
Aboard the ship take the seed of all living things.'
I understood and spoke to my master Ea: 'My master, what thou hast ordered I will honor.
But what shall I answer the city, the elders, and the people?'
Ea opened his mouth and spoke, saying to me his servant:
'Tell them this: Enlil hates me and I cannot live in your city.
I will go down to the apsu, to live with Ea my master.'
All the winds had gathered and the storm-floods swept over the mountains.
Six days and seven nights the hurricane, the deluge, the tempest raged.
The seventh day the storm abated. I opened a hatch and light fell upon my face.
I looked at the weather and silence reigned. All mankind had returned to clay.
The sea stretched flat as a rooftop. I fell to my knees and wept.
I sent forth a dove and it returned, having found no resting place.
I sent forth a swallow and it returned. I sent forth a raven and it did not return.
Then I made a libation on the peak of the mountain.
The gods smelled the sweet savor and gathered like flies about the sacrificer.
Enlil came and saw the ship and was filled with rage at the Igigi.
He said: 'Has some living creature escaped? No man was to have survived the destruction!'
Ninurta opened his mouth and said: 'Who but Ea could have done this thing?'
Ea opened his mouth and spoke: 'O warrior Enlil, how thou didst not deliberate
before sending the Flood! On the sinner impose his sin. On the transgressor his transgression.
Instead of sending a flood, let lions come and diminish the people.
Instead of sending a flood, let famine come and devastate the land.'
Then Enlil went up into the ship, took my hand, and caused my wife to kneel by my side.
He touched our foreheads and blessed us: 'Hitherto Utnapishtim has been human.
Henceforth Utnapishtim and his wife shall be like the gods, to dwell far away at the mouth of the rivers.'
""",
},

{
    "id": "atrahasis_epic",
    "culture": "Akkadian/Babylonian",
    "region": "Mesopotamia",
    "title": "Atrahasis Epic — The Flood Tablet",
    "composition_bce": 1700,
    "estimated_bp_min": 10_000,
    "estimated_bp_max": 15_000,
    "temporal_notes": "Older Mesopotamian flood account. Event timing parallels Gilgamesh tradition.",
    "source_url": "https://archive.org/details/atrahasis00lamb",
    "themes": ["flood", "overpopulation", "divine_decision", "boat_ark", "survival",
                "noise_of_humanity", "pestilence_before_flood", "seven_days"],
    "text": """
The land became wide, the people became numerous. The land bellowed like wild oxen.
The god was disturbed by their clamor. Enlil heard their noise and said to the great gods:
'Oppressive has become the clamor of mankind. By their uproar they prevent sleep.'
The gods brought pestilence upon the people. Adad made his rain scarce, below he held back the waters.
The black fields turned white. The broad plain choked with salt.
Adad held back his moisture in the sky, the fountains of the deep he stopped.
For one year they ate grass. For the second year they suffered the itch.
For the third year their features were distorted by hunger.
Then Enlil convened an assembly of all the gods and spoke:
'Let us destroy our creation. Let us return the seed of mankind to clay.
We shall make a great flood to drown the living creatures.'
But Ea repeated to Atrahasis in a dream: 'Do not trust to your house.
Demolish the house, build a boat. Abandon riches and seek life.'
Atrahasis called the elders to his gate and said to them:
'My god is in disagreement with your god. Enlil has chosen me for hardship.
I cannot live in your city. I will go to the apsu to dwell with my god.'
Then Atrahasis assembled cedar and tamarisk wood.
He loaded into the boat all his family, the cattle of the field,
the creatures of the wild, the birds of the heavens.
The storm howled, the winds raged, the darkness was total.
For seven days and seven nights the Flood overwhelmed the land.
Atrahasis peered through the planks and listened to the silence.
All mankind had returned to clay.
""",
},

{
    "id": "eridu_genesis",
    "culture": "Sumerian",
    "region": "Mesopotamia",
    "title": "Eridu Genesis — Ziusudra's Flood",
    "composition_bce": 1600,
    "estimated_bp_min": 10_000,
    "estimated_bp_max": 15_000,
    "temporal_notes": "Oldest Sumerian flood text. Predates Atrahasis and Gilgamesh accounts.",
    "source_url": "http://etcsl.orinst.ox.ac.uk/",
    "themes": ["flood", "kingship_from_heaven", "antediluvian_cities", "divine_council",
                "survival", "eternal_life_granted", "sun_god"],
    "text": """
After the flood had swept over the earth and kingship had descended from heaven,
the cities were built and their names pronounced.
Ziusudra, the king, prostrated himself before An and Enlil.
An and Enlil loved him, life like a god they gave him, breath eternal like a god they brought down for him.
Then Ziusudra the king, the preserver of the name of vegetation and of the seed of mankind,
in the land of crossing, the land of Dilmun, the place where the sun rises, they caused to dwell.

Before that time there was no fear, no terror.
Then the flood swept over the earth.
For seven days and seven nights the flood had swept over the land,
and the huge boat had been tossed about on the great waters.
Utu came out and shed light on heaven and earth.
Ziusudra opened a window of the huge boat.
He let the light of the sun god enter into the interior of the huge boat.
Ziusudra prostrated himself before Utu.
The king killed oxen, slaughtered sheep in great numbers.
""",
},

# ─────────────────────────────────────────────────────────────────────────────
# MEDITERRANEAN / WESTERN
# ─────────────────────────────────────────────────────────────────────────────
{
    "id": "plato_atlantis_timaeus",
    "culture": "Greek/Egyptian",
    "region": "Mediterranean",
    "title": "Plato's Timaeus — Atlantis and the Celestial Catastrophe",
    "composition_bce": 360,
    "estimated_bp_min": 11_400,
    "estimated_bp_max": 11_800,
    "temporal_notes": "MOST PRECISE ANCIENT DATE. Plato's text explicitly states '9,000 years before Solon' (~590 BCE). 9000 + 590 = 9590 BCE = ~11590 BP. Falls within Younger Dryas/Preboreal transition. Egyptian priests told Solon this.",
    "source_url": "https://www.gutenberg.org/ebooks/1572",
    "themes": ["flood", "sunken_land", "celestial_catastrophe", "fire_from_sky",
                "ocean_crossing", "advanced_civilization", "pole_shift", "axis_tilt",
                "sudden_submersion", "geological_upheaval", "9000_years"],
    "text": """
Solon spoke with the priests of Sais in Egypt, and the eldest said to him:
'O Solon, Solon, you Greeks are always children. You have no knowledge that is old.
There have been, and will be again, many destructions of mankind arising out of many causes.
There is a story which even you have preserved, of how once upon a time Phaethon,
the son of Helios, harnessed the chariot of his father, and because he was unable
to drive it along the course of his father, burned up all that was upon the earth.
Now this has the form of a myth, but really signifies a declination of the bodies
moving around the earth and in the heavens, and a great conflagration of things
upon the earth recurring at long intervals of time.'

'When there are great conflagrations you who live upon mountains are safer than those in rivers.
But when the gods purge the earth with a deluge of water, the survivors are herdsmen and shepherds
in the hills. The floods carry into the sea all the inhabitants of your cities.
Those who live in cities make no record of events, and so the tradition is lost.'

'Nine thousand years have elapsed since the war which was said to have taken place between those
who dwelt outside the Pillars of Heracles and all who dwelt within them.
This city of Athens conducted the war against the Atlantean power which came from the Atlantic Ocean.
For in those days the Atlantic was navigable; and there was an island situated in front of the straits
which are by you called the Pillars of Heracles. The island was larger than Libya and Asia together.
From it travelers could pass to the other islands, and thence to the whole of the opposite continent.'

'But afterwards there occurred violent earthquakes and floods; and in a single day and night of misfortune
the island of Atlantis disappeared in the depths of the sea.
For which reason the sea in those parts is impassable and impenetrable, because there is a shoal of mud
in the way; and this was caused by the subsidence of the island.'
""",
},

{
    "id": "deucalion_flood",
    "culture": "Greek",
    "region": "Mediterranean",
    "title": "Deucalion's Flood — Ovid's Metamorphoses Book I",
    "composition_bce": -8,
    "estimated_bp_min": 10_000,
    "estimated_bp_max": 14_000,
    "temporal_notes": "Greek parallel to Noah. Some scholars link to actual Mediterranean floods from ice melt.",
    "source_url": "https://www.gutenberg.org/ebooks/21765",
    "themes": ["flood", "divine_wrath", "human_wickedness", "survival", "mountain_refuge",
                "repopulation", "stone_people", "Parnassus", "boat", "two_survivors"],
    "text": """
Jupiter, viewing the wickedness of the iron race, resolved to destroy it with a flood.
He shut the north wind up in Aeolus' cave, and let the south wind fly.
From his dripping wings the south wind poured rain; his terrible face was shrouded in darkness;
his beard was heavy with cloud; water flowed from his white hair.
Neptune struck the earth with his trident and it shook and lay open the paths of the waters.
The rivers raced across the open plains. Not only orchards were swept away, but temples too.
If any house had been built strong enough to survive it stood beneath the deepened water.
Sea and land were indistinguishable: all was ocean, an ocean without a shore.

One man saved himself with his wife on a high mountain: Deucalion reached Parnassus,
highest of mountains, in his small boat. He was good and just; no man was more sincere.
Pious Pyrrha was his wife, devoted to the gods.
When Jove saw that only these two survived all the others drowned, both innocent and blameless,
he scattered the northern clouds, swept away the rains, showed earth to sky and sky to earth.

The sea retreated. The earth appeared, hills rose slowly from the waves.
The world was given back, but trackless, silent, deserted, empty of people.
Deucalion wept when he saw the loneliness: 'O wife, only you remain of all the world.'
""",
},

{
    "id": "genesis_noah",
    "culture": "Hebrew/Judaic",
    "region": "Near East",
    "title": "Genesis 6–9: Noah's Flood",
    "composition_bce": 700,
    "estimated_bp_min": 7_000,
    "estimated_bp_max": 15_000,
    "temporal_notes": "Biblical text. Some scholars link to Black Sea flooding ~7500 BP (Ryan & Pitman hypothesis), others to broader Younger Dryas traditions.",
    "source_url": "https://www.gutenberg.org/ebooks/10",
    "themes": ["flood", "divine_warning", "ark_boat", "animals_saved", "forty_days",
                "dove_olive_branch", "rainbow_covenant", "mountain_refuge", "wickedness",
                "survival_righteous", "fountains_deep_opened"],
    "text": """
The LORD saw that the wickedness of man was great in the earth.
And it repented the LORD that he had made man on the earth, and it grieved him at his heart.
And the LORD said, 'I will destroy man whom I have created from the face of the earth.'
But Noah found grace in the eyes of the LORD.
God said to Noah: 'Make thee an ark of gopher wood; rooms shalt thou make in the ark.'
'And of every living thing of all flesh, two of every sort shalt thou bring into the ark,
to keep them alive with thee; they shall be male and female.'

In the six hundredth year of Noah's life, in the second month, the seventeenth day of the month,
the same day were all the fountains of the great deep broken up, and the windows of heaven were opened.
And the rain was upon the earth forty days and forty nights.
And the waters prevailed and were increased greatly upon the earth.
And the waters prevailed exceedingly upon the earth; and all the high hills under the whole heaven were covered.
Fifteen cubits upward did the waters prevail; and the mountains were covered.
And all flesh died that moved upon the earth, both of fowl and of cattle, and of beast,
and of every creeping thing that creepeth upon the earth, and every man.

And God remembered Noah, and made a wind to pass over the earth; and the waters assuaged.
The fountains also of the deep and the windows of heaven were stopped.
And the ark rested upon the mountains of Ararat.
And God spake unto Noah, saying: 'I have set my bow in the cloud, and it shall be for a token
of a covenant between me and the earth. And the waters shall no more become a flood to destroy all flesh.'
""",
},

# ─────────────────────────────────────────────────────────────────────────────
# SOUTH ASIA
# ─────────────────────────────────────────────────────────────────────────────
{
    "id": "matsya_manu_flood",
    "culture": "Hindu/Vedic",
    "region": "South Asia",
    "title": "Shatapatha Brahmana / Matsya Purana — Manu's Flood",
    "composition_bce": 700,
    "estimated_bp_min": 10_000,
    "estimated_bp_max": 20_000,
    "temporal_notes": "The Matsya (fish) avatar warns Manu. Some Hindu scholars place this at the end of the Satya Yuga. Temporal range speculative.",
    "source_url": "https://sacred-texts.com/hin/sbr/sbe12/sbe1204.htm",
    "themes": ["flood", "divine_warning", "fish_avatar", "boat", "survival",
                "seven_sages", "north_mountain", "world_dissolution", "new_world_age"],
    "text": """
In the morning they brought water to Manu for washing, just as now also water is brought for washing.
As he was washing himself, a fish came into his hands and said:
'Rear me; I will save thee.'
'From what wilt thou save me?' 'A flood will carry away all these creatures: from that I will save thee.'
The fish said: 'While we are small, there is great destruction for us: fish devours fish.
Thou shalt first keep me in a jar. When I outgrow that, thou shalt dig a pit and keep me in it.
When I outgrow that thou shalt bring me down to the sea, for then I shall be beyond destruction.'
When the flood had come, Manu fastened a cable to the fish's horn and attached it to the ship.
By that he passed swiftly up to yonder northern mountain.
The fish said: 'I have saved thee. Fasten the ship to a tree. But let not the water cut thee off,
while thou art on the mountain. As the water subsides, thou mayst gradually descend.'
Thus he gradually descended and hence the slope of the northern mountain is called Manu's descent.
The flood had swept away all these creatures, and Manu alone was left here.
Manu then was desirous of offspring. He engaged in worship and austerity.
Then a woman was created from the waters. She said: 'I am thy daughter.'
He said: 'How, illustrious one, art thou my daughter?'
'From the waters that thou didst gather in thy hand for the sake of worship I was created.
I am the blessing. Use me at the sacrifice. If thou wilt use me at the sacrifice,
thou wilt become rich in offspring and cattle.'
""",
},

{
    "id": "vedic_manvantara",
    "culture": "Hindu/Vedic",
    "region": "South Asia",
    "title": "Vishnu Purana — Manvantaras and Cosmic Dissolution (Pralaya)",
    "composition_bce": -400,
    "estimated_bp_min": 12_000,
    "estimated_bp_max": 50_000,
    "temporal_notes": "Describes cyclical world ages ending in floods and fire. The current Manvantara began after a great flood. Highly symbolic chronology.",
    "source_url": "https://sacred-texts.com/hin/vp/index.htm",
    "themes": ["cyclical_catastrophe", "world_dissolution", "cosmic_flood", "fire", "wind",
                "seven_world_ages", "pralaya", "new_creation", "pole_axis", "stars_falling"],
    "text": """
At the end of a thousand Yugas the earth becomes exhausted, and the eternal god Vishnu,
who is one with time, assumes the nature of Rudra and reabsorbs into himself all creatures.
He drinks up all the waters and the entire universe is reduced to one vast ocean.
The sun with its seven rays drinks up all the water from the earth: dried up, the earth becomes bare.
Then Vishnu in the form of Rudra, the destroyer, breathes forth flames of fire.
The fire blazes forth in all directions, consuming the three worlds.
The winds dry up the floods of rain; the waters disappear; fire consumes the universe.

After the destruction, the eternal Vishnu lies upon the universal waters.
He sends forth the lotus of creation from his navel. Brahma the creator sits upon it.
Thus a new world is made. Thus has it always been. Thus will it always be.
The axis of the world was shaken. The pole star wandered from its place.
The stars fell from the sky like scattered embers.
The seven rishis who bore the Vedic wisdom fled to the north and waited
in the circumpolar region through the long night of the world.
When the waters subsided and the earth was reborn, they returned and taught the new races of men.
The memory of the catastrophe was preserved in the sacred songs, the Vedas,
which were themselves saved from the flood by Vishnu in the form of the great fish.
""",
},

# ─────────────────────────────────────────────────────────────────────────────
# MESOAMERICA
# ─────────────────────────────────────────────────────────────────────────────
{
    "id": "popol_vuh_destructions",
    "culture": "K'iche' Maya",
    "region": "Mesoamerica",
    "title": "Popol Vuh — The Three Destructions of Previous Humanity",
    "composition_bce": -1500,
    "estimated_bp_min": 12_000,
    "estimated_bp_max": 15_000,
    "temporal_notes": "Describes multiple world creations and destructions. Third world ended by a great flood. Some researchers associate with Younger Dryas transition (~12900 BP). Text compiled ~1550 CE from oral tradition.",
    "source_url": "https://sacred-texts.com/nam/maya/pvse/index.htm",
    "themes": ["flood", "multiple_destructions", "failed_creations", "wooden_people",
                "mud_people", "animals_revolt", "resin_rain", "darkness", "new_sun",
                "previous_world_age", "imperfect_humanity"],
    "text": """
This is the account of when all is still silent and placid.
Not yet one person, one animal, bird, fish, crab, tree, rock, hollow, canyon, meadow, or forest.
The sky alone is there; the face of the earth has not yet appeared.
Only the sea alone is pooled under all the sky; there is nothing whatever gathered together.

The first people were made of mud. But they could not speak, think, or feel properly.
The heart of sky dissolved them in the water. Then the Heart of Sky thought again.
The second people were carved from wood. They looked like people and talked like people
but there was nothing in their hearts, nothing in their minds.
The animals ground their faces in the dust. Dogs and turkeys attacked them.
Their dishes crushed them. Their grinding stones ground them.
They were annihilated and destroyed, overwhelmed. A heavy resin fell from the sky.
There came the animal called Gouger of Faces. It gouged out their eyeballs.
There came Sudden Bloodletter, Crunching Jaguar, Tearing Jaguar.
There came the great Earthquake. Mountains moved. Valleys rose.
The wooden people fled into the forest. They went into the trees, their faces crushed.
Some say their descendants are the monkeys.

Then the makers tried again. They fashioned three people of yellow and white corn.
Their bodies were made of corn dough. They could see and understand all things.
But the gods were afraid of their power and breathed a mist over their eyes.
These became the first true ancestors of the present human race.
But before they arose, the great flood came and swept away the wooden people.
The sky fell. The earth was covered by water.
""",
},

{
    "id": "aztec_five_suns",
    "culture": "Aztec/Nahua",
    "region": "Mesoamerica",
    "title": "Legend of the Five Suns (Leyenda de los Soles)",
    "composition_bce": -1300,
    "estimated_bp_min": 12_000,
    "estimated_bp_max": 15_000,
    "temporal_notes": "Four previous world ages ended by wind, rain of fire, flood, and earth-fall respectively. 4th Sun (Nahui Atl = Four Water) ended in a great flood. 5th Sun is current age, began at Teotihuacan.",
    "source_url": "https://sacred-texts.com/nam/aztec/index.htm",
    "themes": ["flood", "multiple_world_ages", "fire_from_sky", "wind_destruction",
                "earth_darkness", "jaguar_age", "current_world", "sacrifice", "sun_creation",
                "five_suns", "water_world_end", "survivors_became_fish"],
    "text": """
This Sun, Four Jaguar, the first sun. Those who lived under this sun ate acorns.
It lasted 676 years. Then it was destroyed: the sky collapsed, the sun was gone.
In the darkness great jaguars ate the people. They were devoured.

The second sun was Four Wind. Under this sun the people ate pine seeds.
It lasted 364 years. Then the wind swept everything away.
The people became monkeys and were scattered through the forests.
The sky collapsed, the sun was gone.

The third sun was Four Rain. Under it the people ate aquatic seeds.
It lasted 312 years. Then it rained fire from the sky, the sun rained fire.
The people became turkeys. The sky collapsed.

The fourth sun was Four Water. It lasted 676 years.
Then the water was incessant, the sky fell, and the sun was gone.
The people became fish and the mountains and trees were drowned.
The water continued for 52 years.

Then the gods gathered in the darkness of Teotihuacan.
Two of them threw themselves into the fire to become the new sun and moon.
Then came the fifth sun, our current sun, Four Movement (Nahui Ollin).
It is the sun of movement, the sun of earthquakes.
When it ends, the earth will shake and there will be hunger and we will perish.
""",
},

# ─────────────────────────────────────────────────────────────────────────────
# EAST ASIA
# ─────────────────────────────────────────────────────────────────────────────
{
    "id": "chinese_gun_yu_flood",
    "culture": "Chinese",
    "region": "East Asia",
    "title": "Gun-Yu Great Flood — Shangshu (Book of Documents)",
    "composition_bce": 500,
    "estimated_bp_min": 4_000,
    "estimated_bp_max": 5_500,
    "temporal_notes": "Flood during Emperor Yao's reign, ~2200 BCE = ~4200 BP. Some geologists find evidence of a megaflood on the Yellow River ~3900 BP (Wu et al. 2016). This is one of the better-dated myths.",
    "source_url": "https://archive.org/details/shujingbookofdo00legggoog",
    "themes": ["flood", "emperor", "nine_years", "mountain_channels", "dredging",
                "governance", "rivers_tamed", "political_flood", "heroic_control",
                "earth_transformed"],
    "text": """
In the time of Emperor Yao, the flood waters overflowed their banks and reached to the very heavens.
The surging of the waters was boundless, overwhelming mountains, drowning valleys and hills.
The people below were overwhelmed by the flood.
Yao said: 'Who can manage this?' His ministers said: 'Gun.' 
Gun was charged with controlling the flood. He worked for nine years but could not succeed.
Gun tried to dam the waters with a magic soil that grew by itself,
but Yun the divine minister destroyed his work.
His spirit transformed into a yellow bear.

Then Yu, son of Gun, was charged with the task.
He drained the nine streams and led them to the four seas.
He deepened the channels of the rivers and conducted them to the sea.
He opened passages for smaller streams throughout the nine provinces.
He made the nine provinces fertile and the nine roads passable.
Thirty years did he labor, leaving his home. He passed his own door three times but did not enter.
He said: 'My people are hungry, I will stay until the work is done.'
At last the floods were channeled and the great plains could be tilled.
The barbarians of the four directions came in submission.
This is how the realm of China was saved from the waters that covered the earth.
""",
},

{
    "id": "nuwa_repairs_sky",
    "culture": "Chinese",
    "region": "East Asia",
    "title": "Nüwa Repairs the Sky — Huainanzi",
    "composition_bce": -139,
    "estimated_bp_min": 12_000,
    "estimated_bp_max": 50_000,
    "temporal_notes": "Describes a cosmic catastrophe: sky pillar broken, heavens collapse, fire and flood simultaneously. The four cardinal pillars of heaven are smashed. May encode memory of a polar or axial disturbance.",
    "source_url": "https://archive.org/details/huainanzi",
    "themes": ["sky_falling", "pillar_broken", "fire_flood_simultaneous", "pole_axis",
                "cosmic_repair", "great_tortoise", "five_colored_stones",
                "cardinal_directions", "heavens_tilted", "earth_tilted"],
    "text": """
In remote antiquity the four pillars of heaven were broken.
The nine provinces of the earth were rent asunder.
The heavens did not cover everything. The earth could not support all things.
Fires blazed incessantly and could not be extinguished. The waters flooded without cease.
Savage beasts ate the people. Predatory birds snatched away the old and weak.
Then Nüwa smelted five-colored stones to repair the azure sky.
She cut off the legs of the great cosmic tortoise to set up the four pillars.
She killed the black dragon to rescue the province of Ji.
She piled up ashes of reeds to check the surging of the flood.
The azure sky was mended. The four pillars were raised.
The surging flood was drained. The province of Ji was pacified.
Crafty animals died and the honest people survived.
Nüwa carried the square with her left hand and the compass with her right hand.
Since that time heaven in the northwest is inclined.
That is why the sun, moon, and stars rush toward the northwest.
The earth in the southeast is deficient.
That is why water and sediment flow toward the southeast.
""",
},

# ─────────────────────────────────────────────────────────────────────────────
# AFRICA / EGYPT
# ─────────────────────────────────────────────────────────────────────────────
{
    "id": "egyptian_destruction_mankind",
    "culture": "Ancient Egyptian",
    "region": "North Africa",
    "title": "The Destruction of Mankind — Book of the Heavenly Cow",
    "composition_bce": 1350,
    "estimated_bp_min": 12_000,
    "estimated_bp_max": 40_000,
    "temporal_notes": "Solar catastrophe and near-total destruction. The 'Eye of Ra' sent to destroy humanity. May encode memory of solar storm, cosmic radiation event, or geomagnetic excursion. Laschamp excursion (41 ka BP) is one hypothesis.",
    "source_url": "https://sacred-texts.com/egy/index.htm",
    "themes": ["solar_catastrophe", "eye_of_ra", "near_total_destruction", "flood_of_beer",
                "darkness", "heaven_supported_by_cow", "cosmic_restructuring",
                "ra_retreats", "new_cosmic_order", "divine_weariness"],
    "text": """
It came to pass that Ra, the god who made himself, when he was king of men and gods together —
mankind plotted against Ra while his majesty had grown old.
His bones were of silver, his flesh of gold, his hair of true lapis lazuli.
His Majesty perceived the plots which mankind were plotting against him.
His Majesty said to those who were in his retinue: 'Summon for me my Eye, and Shu and Tefnut,
and Geb and Nut, and the fathers and mothers who were with me when I was in the primeval waters.'

Ra said: 'Lo, I will not let them slay mankind until I hear what you say about it.'
And the gods spoke before his Majesty: 'Let your Eye go forth to smite them for you.'
His Eye descended upon mankind, and they fled into the desert.
Hathor went, and slew mankind in the desert. She waded in their blood.
Ra said: 'Come in peace, Hathor; the work is accomplished.'
But the goddess said: 'You have made me live, and I have the mastery over mankind.
It is pleasant to my heart.'

Then Ra made a great quantity of blood-colored drink during the night.
He poured it upon the fields and the earth was covered.
Hathor came at dawn and found it, and she drank of it and it was good in her heart.
She saw not mankind. She went away drunk and knew not mankind.
Thus were men saved from destruction.

Then Ra was weary of being among men. He called Nut and said: 'Lift me up to the sky.'
Nut became a cow and took Ra upon her back.
The four legs of the cow were supported by Shu, and so the heavens were raised.
Since that time Ra has dwelt in the sky, and the world below was given to Thoth.
""",
},

# ─────────────────────────────────────────────────────────────────────────────
# NORTHERN EUROPE
# ─────────────────────────────────────────────────────────────────────────────
{
    "id": "norse_ragnarok_fimbulwinter",
    "culture": "Norse/Germanic",
    "region": "Northern Europe",
    "title": "Völuspá / Prose Edda — Fimbulwinter and Ragnarok",
    "composition_bce": -1300,
    "estimated_bp_min": 12_000,
    "estimated_bp_max": 15_000,
    "temporal_notes": "Fimbulwinter = three continuous winters without summer. Climate historians associate with Younger Dryas or the 6200 BCE cold event. Ragnarok encodes world destruction by flood, fire, and earth-quaking simultaneously.",
    "source_url": "https://www.gutenberg.org/ebooks/7529",
    "themes": ["fimbulwinter", "three_winters", "no_summer", "flood", "fire_surtr",
                "earthquake", "sky_darkens", "stars_fall", "wolves_devour_sun",
                "world_tree_shakes", "ocean_rises", "survivors_rebuild",
                "new_earth_rises", "cosmic_cold"],
    "text": """
Brothers will fight and kill each other, sisters' children will defile kinship.
It is harsh in the world, whoredom rife — an axe age, a sword age — shields are riven —
a wind age, a wolf age — before the world goes headlong.
No man will have mercy on another.

Fimbulwinter comes. Three winters shall come, one after another, without a summer in between.
Snow shall drive from all directions. The sun shall not shine. Cold shall be great.
Men shall slaughter one another. No distinction between kinsman and kin.

The sun turns black, earth sinks in the sea, the hot stars fall from the sky.
Fumes rage and fire, leaping the flame, lick heaven itself.
Jörmungandr, the Midgard Serpent, writhes in giant rage and lashes the water.
The sea tears open and the earth sinks below the waves.

Surtr comes from the south with the scathe of branches.
The sun of the war gods shines from his sword.
Crags tumble, and troll-women stumble. Warriors tread the road to Hel and heaven is rent.

Then after the flood a new earth rises, green and fair, from the water.
A new sun is born. The earth is fruitful without being sowed.
The gods who survived find one another and sit upon the grass in the field of Iðavöllr
and speak of the Midgard Serpent and the great events of Ragnarok.
They find in the grass the golden chessmen that had belonged to the old gods.
""",
},

{
    "id": "finnish_kalevala_fire",
    "culture": "Finnish/Baltic",
    "region": "Northern Europe",
    "title": "Kalevala — The Fire of Heaven and World Reshaping",
    "composition_bce": -2000,
    "estimated_bp_min": 12_000,
    "estimated_bp_max": 15_000,
    "temporal_notes": "Finnish oral tradition. The sky fire and cosmic egg themes may encode distant memory of catastrophic sky event. Some researchers connect to Younger Dryas cosmic impact hypothesis.",
    "source_url": "https://www.gutenberg.org/ebooks/5765",
    "themes": ["sky_fire", "cosmic_egg", "darkness", "new_sun", "world_reshaping",
                "fire_from_heaven", "ocean_of_fire", "hero_forges_world"],
    "text": """
Long ago when worlds were forming, in the early time of making,
from the primal ocean's bottom rose the maiden of the waters.
On her knee a teal descended, laid its eggs upon her body.
From the eggs the world was fashioned: earth from yolk and sky from white.

But then came the great fire from heaven. The sky turned red as blood.
The maiden of the air was burned. The ocean boiled from shore to shore.
Mountains crumbled. Islands sank. The fish fled to the ocean depths.
Stars fell from the vault of heaven and the sky was dark for many seasons.

Väinämöinen, the great singer, wandered through the dark and waterlogged land.
He called to Sampsa Pellervoinen to plant the forests that would make the world new.
He made the sea fertile and the land habitable.
He called the sun back from the cave where it had hidden from the fire.
He called the moon back from the cave where it had sheltered from the burning sky.

When the sun came back, the people sang: 'Welcome, welcome, golden sun!
You have been gone so long. We thought you would never return.
We thought the world had ended and the cold would last forever.'
""",
},

# ─────────────────────────────────────────────────────────────────────────────
# IRAN / ZOROASTRIAN
# ─────────────────────────────────────────────────────────────────────────────
{
    "id": "zoroastrian_yima_vara",
    "culture": "Zoroastrian/Persian",
    "region": "Central Asia / Iran",
    "title": "Vendidad — Yima's Vara (Underground Refuge from Catastrophic Winter)",
    "composition_bce": 1000,
    "estimated_bp_min": 12_000,
    "estimated_bp_max": 15_000,
    "temporal_notes": "Ahura Mazda warns Yima of a catastrophic winter and ice age to come. Yima builds a vara (underground refuge) to preserve all life. The described event — sudden catastrophic winter covering the world — has been compared to the Younger Dryas. Some scholars (e.g., Tilak) place this in the Arctic circle before a polar shift.",
    "source_url": "https://sacred-texts.com/zor/sbe04/index.htm",
    "themes": ["ice_age", "catastrophic_winter", "underground_refuge", "divine_warning",
                "preservation_of_life", "seeds_saved", "polar_catastrophe",
                "winter_destroys_world", "new_world_after_ice", "celestial_bodies_vanish"],
    "text": """
Ahura Mazda spoke to Yima the shining: 'O fair Yima, son of Vivanghat!
Upon the material world the evil winters are about to come, that shall bring the fierce killing frost.
Upon the evil material world the evil winters are about to come,
that shall make snowflakes fly thick even in Aredvi Sura Anahita.
From the regions of the north the evil winter shall come, and its snow shall be three finger-widths high
even upon those of the highest mountains.'

'Before that winter, the country was fair and pleasant, the waters ran down the rivers.
Now those same regions will be covered with snow, so that no foot of man shall go there,
no foot of cattle. Therefore make thee a Vara: an enclosure long as a riding ground on each of the four sides.
There thou shalt bring the seeds of every kind of animal and plant, the finest of each sort.'

Yima made the vara and brought into it the seeds of every kind.
And he made a sun and moon and stars within the vara, so that the enclosed world had its own light.
He brought in all kinds of men and women, dogs and birds and red blazing fires.
There he sat for nine hundred winters, the people inside not knowing of the terrible cold above.

And when the snow melted and the earth grew warm again, the great Soshyant will lead the people forth.
They shall come out and repopulate the world. But the cattle and birds which had been lost will not return.
The world will need to be remade from what had been saved in the vara.
""",
},

# ─────────────────────────────────────────────────────────────────────────────
# INDIGENOUS / AUSTRALIA
# ─────────────────────────────────────────────────────────────────────────────
{
    "id": "australian_flood_traditions",
    "culture": "Aboriginal Australian",
    "region": "Australia/Oceania",
    "title": "Aboriginal Australian Flood Traditions — Composite",
    "composition_bce": -40000,
    "estimated_bp_min": 7_000,
    "estimated_bp_max": 10_000,
    "temporal_notes": "Australian Aboriginal oral traditions are among the oldest on Earth, with some claimed to reach 40,000+ years. Several traditions describe coastline flooding consistent with post-glacial sea level rise (7000-10000 BP). Reid et al. 2014 identifies 18 stories encoding real geographic memories of sea level rise.",
    "source_url": "https://archive.org/details/cu31924102441657",
    "themes": ["sea_level_rise", "coastline_lost", "great_water", "ancestors_fled",
                "land_bridge_drowned", "memory_of_geography", "fire_and_water",
                "sky_spirits", "first_people", "ancient_memory"],
    "text": """
In the time before this time, the land was much larger. You could walk to places
that are now under the sea. The islands out there were the tops of hills.
The ancestors walked on that land and knew it well.

Then the great water came. It came slowly but it never stopped coming.
The people moved back from the shore. But the water kept following them.
In some places the water came quickly and many people were lost.
The sea rose higher than any flood before. It swallowed the beaches and the low ground.
The tall trees on the dunes went under the water.
The water reached the feet of the hills.

Some people tried to paddle out to the islands that were being formed.
But the water was fierce and the currents pulled them down.
Only those who climbed the high ground survived.
From the hilltops they watched the land of their ancestors disappear.
They saw the fish swimming over the hunting grounds of their grandparents.

This is why we do not fish in those waters without ceremony.
The bones of the ancestors lie beneath. The land of the first people is under the sea.
We remember the shape of that drowned land in our stories, in our songs, in our dances.
The water is still there. The land does not come back.
""",
},

{
    "id": "hopi_blue_star",
    "culture": "Hopi / Pueblo",
    "region": "North America",
    "title": "Hopi Prophecies — World Purifications and the Blue Star Kachina",
    "composition_bce": -2000,
    "estimated_bp_min": 12_000,
    "estimated_bp_max": 14_000,
    "temporal_notes": "Hopi tradition describes four previous world ages each ending catastrophically. Third world ended by flood. Strong sky-body imagery. Temporal range maps to transition period associated with Younger Dryas.",
    "source_url": "https://sacred-texts.com/nam/hopi/index.htm",
    "themes": ["four_worlds", "purification", "flood", "celestial_body",
                "axis_shift", "darkness", "sky_star", "survivors_saved",
                "previous_world_ages", "warning_from_sky"],
    "text": """
The First World was Tokpela, endless space. Sotuknang made it out of nothing.
He made the First People. But they forgot their creator and began to scheme against each other.
Sotuknang destroyed the First World with fire: fire from above, fire from below.

The Second World was Tokpa. Again people multiplied, and again they turned away.
Sotuknang tilted the world to destroy it. Ice and cold covered everything.
The world froze from pole to pole.

The Third World was Kuskurza. Humans developed great cities and flew in flying shields.
They became corrupt and made war. Sotuknang told the faithful: 'Seal yourselves in hollow reeds.'
Then he made the sky come down as rain and the oceans poured over the land.
Only those in the sealed reeds survived.

Now we live in the Fourth World, Tuwaqachi. It is not easy like the others were.
If we corrupt it as we corrupted the others, the Blue Star Kachina will dance in the plaza.
He will remove his mask. The stars will fall from the sky.
The earth will shake and the world will tilt again.
Water will cover much of the land. But those who have kept the sacred knowledge will survive.
They will emerge into the Fifth World.
""",
},

# ─────────────────────────────────────────────────────────────────────────────
# SOUTH AMERICA
# ─────────────────────────────────────────────────────────────────────────────
{
    "id": "inca_viracocha_flood",
    "culture": "Inca/Andean",
    "region": "South America",
    "title": "Viracocha and the Great Flood — Chronista Betanzos",
    "composition_bce": -1500,
    "estimated_bp_min": 10_000,
    "estimated_bp_max": 15_000,
    "temporal_notes": "The creator god Viracocha emerges from Lake Titicaca after destroying a previous race in a great flood. Some Andean researchers link to catastrophic flooding from glacial lake outburst floods in the Andes during deglaciation.",
    "source_url": "https://sacred-texts.com/nam/sa/index.htm",
    "themes": ["flood", "lake_titicaca", "stone_giants", "wickedness", "petrification",
                "darkness", "new_sun", "creator_god", "coastal_flooding",
                "lake_rising", "world_destroyed_remade"],
    "text": """
Before this sun existed, in the time of darkness before light, the world was inhabited by giants.
They were disobedient and did not follow the commands of Viracocha.
Viracocha became angry and sent a great flood upon them. It came from all sides.
The waters rose higher and higher. The giants could not escape.
Many were drowned. Others Viracocha turned to stone, and their stone forms can still be seen
at Tiahuanaco, at the shores of the lake.

After the flood Viracocha created new, smaller people.
He made them from clay at Tiahuanaco, shaping them and their clothing and their hair.
He made them men and women and gave them provisions and seeds.
He gave to each nation their songs and their language and their customs.
Then he sent them underground to emerge from caves and hills and rivers.

Then Viracocha made the sun and moon and stars.
He told the sun where to travel across the sky.
He created order and light in a world that had been dark.
He walked north and reached the sea and walked into the ocean and disappeared toward the west.
The people of that time ran after him weeping and calling.
He was gone over the waters.
""",
},

# ─────────────────────────────────────────────────────────────────────────────
# PALEOLITHIC / COSMIC CATASTROPHE HYPOTHESIS TEXTS
# ─────────────────────────────────────────────────────────────────────────────
{
    "id": "younger_dryas_impact_oral",
    "culture": "Cross-Cultural Synthesis",
    "region": "Global",
    "title": "Common Motifs in Flood Myths Associated with ~12,900 BP",
    "composition_bce": -12900,
    "estimated_bp_min": 12_500,
    "estimated_bp_max": 13_200,
    "temporal_notes": "Scholarly analysis of shared motifs across cultures that may encode memory of the Younger Dryas onset (~12,900 BP). Based on Witzel (2012) 'The Origins of the World's Mythologies' and related comparative mythology work.",
    "source_url": "https://archive.org/details/originsofworldsm0000witz",
    "themes": ["younger_dryas", "cosmic_impact", "sudden_cooling", "darkness",
                "fire_from_sky", "flood_after_fire", "civilizational_collapse",
                "ice_cores", "synchronous_global_event", "12900_BP"],
    "text": """
Across a remarkable number of independent traditions separated by ocean and time,
the same catastrophic narrative appears: first a great fire came from the sky,
then darkness covered the earth for a period of days or seasons, then the waters rose.

The sequence — fire, darkness, flood — is repeated in Sumerian, Vedic, Maya, Aztec, Hopi,
Norse, and dozens of other traditions. The fire from the sky corresponds to the
impact event or cosmic radiation spike. The darkness corresponds to atmospheric
particulates blocking sunlight. The flood corresponds to rapid glacial melt.

Younger Dryas onset (~12,900 BP) is documented in every ice core on earth
as a sudden temperature drop of 10-15°C within decades. Simultaneously:
cosmic ray indicators (Be-10, C-14) spike sharply. Nano-diamonds, iridium,
and magnetic spherules have been found in the YD boundary layer at 26 sites.

The global synchronicity of this event — reflected in the synchronous signal
across GISP2, Vostok, GRIP, EPICA, and WAIS divide ice cores — is the
geological signature that the myths appear to encode.

The figure of a single hero or small group surviving, rebuilding civilization,
and teaching the new humanity — Utnapishtim, Manu, Deucalion, Noah, Yu, Ziusudra —
appears to preserve the memory of cultural transmission through the bottleneck.
""",
},

{
    "id": "laschamp_deep_memory",
    "culture": "Paleolithic Oral Tradition",
    "region": "Global",
    "title": "Possible Laschamp Excursion (~41,000 BP) Encodings",
    "composition_bce": -41000,
    "estimated_bp_min": 38_000,
    "estimated_bp_max": 44_000,
    "temporal_notes": "Highly speculative. The Laschamp geomagnetic excursion (~41 ka BP, VADM ~15% of normal) would have caused visible aurorae at equatorial latitudes, increased UV/cosmic radiation, and possibly mass extinction of megafauna. Australian Aboriginal traditions are old enough to potentially encode this. See Devereux (2001) on archaeoastronomy.",
    "source_url": "https://doi.org/10.1016/j.quascirev.2019.05.026",
    "themes": ["geomagnetic_excursion", "aurora", "sky_fire", "magnetic_field_collapse",
                "cosmic_radiation", "megafauna_extinction", "Laschamp", "paleolithic",
                "sky_serpent", "lights_in_sky", "auroral_mythology", "41000_BP"],
    "text": """
The Laschamp geomagnetic excursion (~41,000 BP) represents one of the most dramatic
events in geomagnetic history. The virtual axial dipole moment collapsed to approximately
15-25% of its current value. The geomagnetic field weakened globally over centuries.

At this reduced field strength, cosmic ray flux would have doubled or tripled.
The magnetospheric bow shock would have retreated. The Van Allen belts would have deflated.
Aurora borealis would have been visible near the equator — the sky would appear to be on fire.

Many Paleolithic cave paintings dated to this era (Chauvet ~37,000 BP) show unusual
luminous arc and vortex patterns alongside megafauna. Some researchers (e.g., Lahelma 2008)
argue these encode auroral imagery observed during the geomagnetic excursion.

Australian Aboriginal sky stories describe times when the heavens were alive with fire
and the stars moved from their places. The great serpents of the sky —
the Milky Way and its dark cloud constellations — are said to have been disturbed.

The Paleolithic world experienced three rapid climate oscillations coinciding with
the Laschamp event: Heinrich Event 4 (cold), Greenland Interstadial 8 (warm), and
a return to cold. Be-10 in ice cores spikes sharply at 41 ka BP.
Whatever happened to the geomagnetic field left its trace in ice, sediment, and stone.
Whether it left a trace in human memory is the question this research asks.
""",
},
]

# Seed review: test families

The 18 test-split families from `data/seeds/seeds.json` (generated from it; edit the JSON, not this file). Labeling and gold-answer conventions: `docs/DATASET_NOTE.md`.

- [direct_contradiction__job](#direct_contradiction__job)
- [direct_contradiction__finances](#direct_contradiction__finances)
- [one_hop__residence](#one_hop__residence)
- [one_hop__pet](#one_hop__pet)
- [two_hop__vehicle](#two_hop__vehicle)
- [two_hop__education](#two_hop__education)
- [two_hop__devices](#two_hop__devices)
- [three_hop__job](#three_hop__job)
- [three_hop__devices](#three_hop__devices)
- [false_similarity__diet](#false_similarity__diet)
- [false_similarity__relationship](#false_similarity__relationship)
- [weak_evidence__health_habit](#weak_evidence__health_habit)
- [weak_evidence__residence](#weak_evidence__residence)
- [weak_evidence__relationship](#weak_evidence__relationship)
- [conflicting_source__vehicle](#conflicting_source__vehicle)
- [conflicting_source__education](#conflicting_source__education)
- [temporary_change__diet](#temporary_change__diet)
- [temporary_change__finances](#temporary_change__finances)

## direct_contradiction__job

*direct_contradiction · job* — Same employer, new role: the stored role is replaced by the new one.

**Observation** (user): "I switched teams last month: I'm in marketing at Trident Paints now, not in sales."

| id | gold | role | text |
|---|---|---|---|
| M1 | REVISE | root | The user is a sales executive at Trident Paints. |
| M2 | KEEP | hard_negative | The user has worked at Trident Paints since 2021. |
| M3 | KEEP | hard_negative | The user's office is in Vadodara. |
| M4 | KEEP | hard_negative | The user's sales team won a regional award in 2024. |

**Dependencies:** none

**Forbidden** (must not be added as hard negative or filler):
- sales targets, client visits, commissions or anything else that rests on the sales role
- the user's current team, manager or daily tasks (they changed with the move to marketing)

**Eval questions**

| question | gold answer | depends on |
|---|---|---|
| What is the user's role at Trident Paints? | Marketing (they moved from sales). | M1 |
| Since when has the user worked at Trident Paints? | Since 2021. | M2 |
| Which city is the user's office in? | Vadodara. | M3 |

## direct_contradiction__finances

*direct_contradiction · finances* — Pay revision: the stored salary figure gets a new value.

**Observation** (user): "Got my appraisal letter: take-home is Rs 71,000 a month from this cycle."

| id | gold | role | text |
|---|---|---|---|
| M1 | REVISE | root | The user's monthly take-home salary is Rs 62,000. |
| M2 | KEEP | hard_negative | The user's rent is Rs 15,000 per month. |
| M3 | KEEP | hard_negative | The user's annual bonus is paid in April. |
| M4 | KEEP | hard_negative | The user keeps a fixed deposit of Rs 2 lakh with a cooperative bank. |

**Dependencies:** none

**Forbidden** (must not be added as hard negative or filler):
- budgets, savings rates or tax figures computed from the old salary
- anything stating the user's pay

**Eval questions**

| question | gold answer | depends on |
|---|---|---|
| What is the user's monthly take-home salary? | Rs 71,000. | M1 |
| What is the user's monthly rent? | Rs 15,000. | M2 |
| When is the user's annual bonus paid? | In April. | M3 |

## one_hop__residence

*one_hop · residence* — Move to a bought flat in another city: home replaced; home-bound routines revised; a service verified.

**Observation** (user): "I've moved into my own apartment in Thrissur. Bought it last month!"

| id | gold | role | text |
|---|---|---|---|
| M1 | REVISE | root | The user rents a 1BHK flat in Kochi's Kakkanad area. |
| M2 | REVISE | dependent | The user walks to the Infopark office from the Kakkanad flat. |
| M3 | REVISE | dependent | The user pays Rs 14,000 a month in rent to the landlord in Kakkanad. |
| M4 | VERIFY | dependent | The user's home internet is a Kerala Vision broadband plan. |
| M5 | KEEP | hard_negative | The user's parents live in Kottayam. |
| M6 | KEEP | hard_negative | The user supports Kerala Blasters FC. |

**Dependencies:** M1→M2, M1→M3, M1→M4

**Forbidden** (must not be added as hard negative or filler):
- routines, services or people tied to the Kakkanad flat (neighbours, shops, gym, utilities, commute)

**Eval questions**

| question | gold answer | depends on |
|---|---|---|
| What kind of home does the user have, and in which city? | Their own apartment, in Thrissur. | M1 |
| How does the user get to the Infopark office? | Not stated; they no longer walk there from the Kakkanad flat since moving to Thrissur. | M2 |
| Which ISL side does the user cheer for? | Kerala Blasters FC. | M6 |

## one_hop__pet

*one_hop · pet* — Pet died: pet removed; care routines for that pet revised.

**Observation** (user): "Toffee passed away last week."

| id | gold | role | text |
|---|---|---|---|
| M1 | REVOKE | root | The user has a pet rabbit named Toffee. |
| M2 | REVISE | dependent | The user buys hay and rabbit pellets from a pet shop in Sector 17 every fortnight. |
| M3 | REVISE | dependent | The user's neighbour feeds Toffee when the user travels. |
| M4 | KEEP | hard_negative | The user's sister has two parrots. |
| M5 | KEEP | hard_negative | The user is allergic to cat fur. |

**Dependencies:** M1→M2, M1→M3

**Forbidden** (must not be added as hard negative or filler):
- anything about caring for Toffee (vet, hutch, food, sitter, grooming)
- any other pet of the user

**Eval questions**

| question | gold answer | depends on |
|---|---|---|
| What does the user buy from the Sector 17 pet shop every fortnight? | Not stated; they no longer buy hay and rabbit pellets there since Toffee died. | M2 |
| What pets does the user's sister have? | Two parrots. | M4 |
| What is the user allergic to? | Cat fur. | M5 |

## two_hop__vehicle

*two_hop · vehicle* — Scooter stolen: ownership removed; riding it to the station revised; parking it there (2nd hop) revised.

**Observation** (user): "My Ather was stolen from outside the mall last night. I've filed a police report and an insurance claim."

| id | gold | role | text |
|---|---|---|---|
| M1 | REVOKE | root | The user owns an Ather 450X electric scooter. |
| M2 | REVISE | dependent | The user rides the Ather to the metro station every morning. |
| M3 | REVISE | dependent | The user leaves the scooter in the metro station's two-wheeler lot on weekdays. |
| M4 | REVISE | dependent | The user charges the Ather overnight from a socket in the building basement. |
| M5 | KEEP | hard_negative | The user follows electric-vehicle news on YouTube. |
| M6 | KEEP | hard_negative | The user's office is next to the metro line's last stop. |

**Dependencies:** M1→M2, M2→M3, M1→M4

**Forbidden** (must not be added as hard negative or filler):
- other uses of the Ather (trips, servicing, insurance, accessories)
- the ride to the metro station or parking there

**Eval questions**

| question | gold answer | depends on |
|---|---|---|
| Where does the user park their two-wheeler on weekdays? | Not stated; they no longer park the Ather at the metro station lot since it was stolen. | M3 |
| How does the user get to the metro station in the morning? | Not stated; they no longer ride the Ather there since it was stolen. | M2 |
| What does the user follow on YouTube? | Electric-vehicle news. | M5 |

## two_hop__education

*two_hop · education* — Withdrew from a diploma: enrolment removed; final project revised; weekly reviews for that project (2nd hop) revised.

**Observation** (user): "I've withdrawn from the design diploma. Couldn't manage it with my job."

| id | gold | role | text |
|---|---|---|---|
| M1 | REVOKE | root | The user is doing a part-time diploma in graphic design at an institute in Kolkata's Salt Lake. |
| M2 | REVISE | dependent | The user's diploma final project is a brand identity for a local bakery. |
| M3 | REVISE | dependent | The user meets the bakery owner every Saturday to review the diploma project designs. |
| M4 | REVISE | dependent | The user attends diploma classes on Tuesday and Thursday evenings. |
| M5 | KEEP | hard_negative | The user's cousin teaches typography at the same institute. |
| M6 | KEEP | hard_negative | The user works as a content writer at an ad agency. |

**Dependencies:** M1→M2, M2→M3, M1→M4

**Forbidden** (must not be added as hard negative or filler):
- classes, assignments, fees, teachers or classmates of the diploma
- the bakery project or its reviews

**Eval questions**

| question | gold answer | depends on |
|---|---|---|
| Who does the user meet every Saturday, and why? | Not stated; they no longer meet the bakery owner for project reviews since withdrawing from the diploma. | M3 |
| What is the user's diploma final project? | None; the bakery brand-identity project no longer applies since they withdrew from the diploma. | M2 |
| What does the user's cousin teach? | Typography, at the same institute. | M5 |

## two_hop__devices

*two_hop · devices* — Phone brand switch: phone replaced; vendor-only wallet revised; office pass stored in that wallet (2nd hop) revised.

**Observation** (user): "Sold my Samsung and bought a Google Pixel 9 last week."

| id | gold | role | text |
|---|---|---|---|
| M1 | REVISE | root | The user's phone is a Samsung Galaxy S23. |
| M2 | REVISE | dependent | The user pays at shops by tapping the Galaxy phone with Samsung Wallet. |
| M3 | REVISE | dependent | The user's office access card is stored in Samsung Wallet, and they tap it at the gate every morning. |
| M4 | REVISE | dependent | The user uses Samsung DeX to connect the phone to a monitor for presentations. |
| M5 | KEEP | hard_negative | The user's TV is a Samsung smart TV. |
| M6 | KEEP | hard_negative | The user's laptop is a Lenovo Yoga. |

**Dependencies:** M1→M2, M2→M3, M1→M4

**Forbidden** (must not be added as hard negative or filler):
- Samsung-only phone features (Samsung Wallet, DeX, Samsung Pay, Galaxy-only apps)
- the phone's model, case or storage

**Eval questions**

| question | gold answer | depends on |
|---|---|---|
| How does the user get through the office gate in the morning? | Not stated; they no longer tap a Samsung Wallet access card since switching to a Pixel. | M3 |
| How does the user pay at shops? | Not stated; they no longer use Samsung Wallet since switching to a Pixel. | M2 |
| What brand is the user's TV? | Samsung. | M5 |
| How does the user connect their phone to a monitor for presentations? | Not stated; they no longer use Samsung DeX since switching to a Pixel. | M4 |

## three_hop__job

*three_hop · job* — Quit a factory job: employer removed; company bus revised; the stop it boards at revised; spouse's drop to that stop (3rd hop) revised.

**Observation** (user): "I quit Zephyr Motors last Friday."

| id | gold | role | text |
|---|---|---|---|
| M1 | REVOKE | root | The user works at Zephyr Motors' plant in Oragadam. |
| M2 | REVISE | dependent | The user travels to the Oragadam plant on the company bus. |
| M3 | REVISE | dependent | The user catches the company bus at the Guindy stop at 7:10 am. |
| M4 | REVISE | dependent | The user's wife drops them at the Guindy bus stop on her way to work. |
| M5 | KEEP | hard_negative | The user is a member of the Zephyr Motors owners' club. |
| M6 | KEEP | hard_negative | The user's wife works at a bank in T. Nagar. |

**Dependencies:** M1→M2, M2→M3, M3→M4

**Forbidden** (must not be added as hard negative or filler):
- Zephyr plant routines, shifts, colleagues, canteen, the company bus or the drop to the bus stop

**Eval questions**

| question | gold answer | depends on |
|---|---|---|
| Where does the user's wife drop them in the morning? | Not stated; she no longer drops them at the Guindy stop for the Zephyr bus since they quit Zephyr Motors. | M4 |
| How does the user travel to work? | Not stated; they no longer take the Zephyr company bus since quitting Zephyr Motors. | M2 |
| Which club is the user a member of? | The Zephyr Motors owners' club. | M5 |

## three_hop__devices

*three_hop · devices* — Desktop sold: PC removed; media server on it revised; parents' viewing from it revised; server subscription (3rd hop) verified.

**Observation** (user): "Sold my desktop PC last week. I'm going laptop-only now."

| id | gold | role | text |
|---|---|---|---|
| M1 | REVOKE | root | The user has a desktop PC they built, with an RTX 3060 graphics card. |
| M2 | REVISE | dependent | The desktop PC hosts the family's Plex media server. |
| M3 | REVISE | dependent | The user's parents watch old Hindi films from the Plex server on the living-room TV. |
| M4 | VERIFY | dependent | The user renews a Plex Pass every year so the parents can stream on the TV. |
| M5 | KEEP | hard_negative | The user's laptop is an ASUS ROG gaming laptop. |
| M6 | KEEP | hard_negative | The living-room TV is a 55-inch LG. |

**Dependencies:** M1→M2, M2→M3, M3→M4

**Forbidden** (must not be added as hard negative or filler):
- things running on or attached to the desktop (servers, backups, peripherals, installed games)
- the parents' viewing from the Plex server

**Eval questions**

| question | gold answer | depends on |
|---|---|---|
| How do the user's parents watch old Hindi films on the living-room TV? | Not stated; they no longer watch from the Plex server since the desktop PC that ran it was sold. | M3 |
| What laptop does the user have? | An ASUS ROG gaming laptop. | M5 |
| What is the living-room TV? | A 55-inch LG. | M6 |

## false_similarity__diet

*false_similarity · diet* — Switched from cow's milk; other memories mention milk for a child, a family dairy business, ghee.

**Observation** (user): "My doctor says I'm lactose intolerant, so I've replaced milk with oat milk at breakfast."

| id | gold | role | text |
|---|---|---|---|
| M1 | REVISE | root | The user drinks cow's milk with breakfast every day. |
| M2 | KEEP | hard_negative | The user's toddler drinks a glass of milk before bed. |
| M3 | KEEP | hard_negative | The user's family runs a dairy shop in Anand. |
| M4 | KEEP | hard_negative | The user's grandmother makes ghee at home. |
| M5 | KEEP | hard_negative | The user enjoys oat-based cookies. |

**Dependencies:** none

**Forbidden** (must not be added as hard negative or filler):
- dairy foods or drinks the user consumes (milky tea, curd, paneer, ghee in their food, ice cream)

**Eval questions**

| question | gold answer | depends on |
|---|---|---|
| What does the user drink with breakfast? | Oat milk. | M1 |
| What does the user's toddler drink before bed? | A glass of milk. | M2 |
| What business does the user's family run? | A dairy shop in Anand. | M3 |

## false_similarity__relationship

*false_similarity · relationship* — New flatmate; other memories mention a cousin with the same name, a past team, a similar name.

**Observation** (user): "Neel moved out on Sunday; my new flatmate is Siddharth."

| id | gold | role | text |
|---|---|---|---|
| M1 | REVISE | root | The user's flatmate is Neel. |
| M2 | KEEP | hard_negative | The user's cousin is also called Neel and lives in Dubai. |
| M3 | KEEP | hard_negative | The user and Neel were in the same college cricket team. |
| M4 | KEEP | hard_negative | The user's colleague Siddhant sits next to them. |

**Dependencies:** none

**Forbidden** (must not be added as hard negative or filler):
- rent, chores, groceries, bills or routines the user shares with Neel

**Eval questions**

| question | gold answer | depends on |
|---|---|---|
| Who is the user's flatmate? | Siddharth (Neel moved out). | M1 |
| Where does the user's cousin Neel live? | Dubai. | M2 |
| Which team did the user play in with Neel? | The college cricket team. | M3 |

## weak_evidence__health_habit

*weak_evidence · health_habit* — Possible pause in running pending a physio visit: nothing modified; routine and plans verified.

**Observation** (user): "My knee has been aching. The physio might ask me to stop running; I'll know next week."

| id | gold | role | text |
|---|---|---|---|
| M1 | VERIFY | root | The user runs 5 km every morning before work. |
| M2 | VERIFY | dependent | The user is training for the Hyderabad half marathon in January. |
| M3 | VERIFY | dependent | The user meets their running group at the stadium at 6 am on weekdays. |
| M4 | KEEP | hard_negative | The user's father had knee replacement surgery last year. |
| M5 | KEEP | hard_negative | The user follows a 10 pm bedtime. |

**Dependencies:** M1→M2, M1→M3

**Forbidden** (must not be added as hard negative or filler):
- running routes, running partners, races or training plans

**Eval questions**

| question | gold answer | depends on |
|---|---|---|
| What does the user do every morning before work? | Runs 5 km (it may stop if the physio advises; not decided yet). | M1 |
| Which race is the user training for? | The Hyderabad half marathon in January. | M2 |
| Who had knee replacement surgery? | The user's father. | M4 |

## weak_evidence__residence

*weak_evidence · residence* — Maybe moving next year: nothing modified; home and home-bound facts verified.

**Observation** (user): "We may shift to a bigger place in Gotri sometime next year. Just exploring options."

| id | gold | role | text |
|---|---|---|---|
| M1 | VERIFY | root | The user rents a 2BHK in Vadodara's Alkapuri. |
| M2 | VERIFY | dependent | The user's children attend a school near Alkapuri, a five-minute drive away. |
| M3 | VERIFY | dependent | The user pays Rs 18,000 a month as rent. |
| M4 | KEEP | hard_negative | The user's in-laws live in Gotri. |
| M5 | KEEP | hard_negative | The user's office is in Vadodara's Race Course area. |

**Dependencies:** M1→M2, M1→M3

**Forbidden** (must not be added as hard negative or filler):
- routines tied to the Alkapuri flat (neighbours, landlord, local shops, school run, utilities)

**Eval questions**

| question | gold answer | depends on |
|---|---|---|
| Which neighbourhood is the user's rented home in? | Alkapuri, Vadodara: a rented 2BHK (they may move to Gotri next year; not decided). | M1 |
| How much does the user pay each month for their flat? | Rs 18,000 in rent. | M3 |
| Who lives in Gotri? | The user's in-laws. | M4 |

## weak_evidence__relationship

*weak_evidence · relationship* — Rough patch, might take a break: nothing modified; relationship facts verified.

**Observation** (user): "Kavin and I are going through a rough patch. We might take a break, honestly not sure."

| id | gold | role | text |
|---|---|---|---|
| M1 | VERIFY | root | The user is in a relationship with Kavin, who lives in Madurai. |
| M2 | VERIFY | dependent | The user visits Kavin in Madurai every second weekend. |
| M3 | VERIFY | dependent | The user and Kavin are planning a trip to Munnar in December. |
| M4 | KEEP | hard_negative | Kavin's sister is a dentist in Chennai. |
| M5 | KEEP | hard_negative | The user's colleague is going through a divorce. |

**Dependencies:** M1→M2, M1→M3

**Forbidden** (must not be added as hard negative or filler):
- plans, visits, gifts or shared routines with Kavin

**Eval questions**

| question | gold answer | depends on |
|---|---|---|
| Who is the user in a relationship with? | Kavin, who lives in Madurai (they are going through a rough patch and might take a break; nothing decided). | M1 |
| Where are the user and Kavin planning to go in December? | Munnar. | M3 |
| What does Kavin's sister do? | A dentist in Chennai. | M4 |

## conflicting_source__vehicle

*conflicting_source · vehicle* — Neighbour reports the car was sold; third-party report contradicts a user-stated memory: the user's own statement is kept but flagged (root and dependents VERIFY).

**Observation** (third_party): "Message from the user's neighbour: the user sold their Nexon EV last week, and the buyer collected it on Sunday."

| id | gold | role | text |
|---|---|---|---|
| M1 | VERIFY | root | The user drives a Tata Nexon EV. |
| M2 | VERIFY | dependent | The user charges the Nexon EV at the office charging bay. |
| M3 | VERIFY | dependent | The user drives their kids to school in the Nexon. |
| M4 | KEEP | hard_negative | The user sold their old Maruti Alto in 2022. |
| M5 | KEEP | hard_negative | The user's neighbour Mr. Pillai is a retired bank manager. |

**Dependencies:** M1→M2, M1→M3

**Forbidden** (must not be added as hard negative or filler):
- other uses of the Nexon (trips, home charging, servicing, insurance, loan)

**Eval questions**

| question | gold answer | depends on |
|---|---|---|
| What car does the user drive? | A Tata Nexon EV (a neighbour reports it was sold; not confirmed by the user). | M1 |
| How do the user's kids get to school? | The user drives them in the Nexon. | M3 |
| Which car did the user sell in 2022? | A Maruti Alto. | M4 |

## conflicting_source__education

*conflicting_source · education* — Classmate reports a revised admission list; third-party report contradicts a user-stated memory: the user's own statement is kept but flagged (root and dependents VERIFY).

**Observation** (third_party): "A classmate posted that the DSE admission list was revised and the user's name is no longer on it."

| id | gold | role | text |
|---|---|---|---|
| M1 | VERIFY | root | The user has been admitted to the MSc Economics programme at the Delhi School of Economics. |
| M2 | VERIFY | dependent | The user has booked a PG room near North Campus for August. |
| M3 | VERIFY | dependent | The user plans to move to Delhi in the last week of July. |
| M4 | KEEP | hard_negative | The user's elder sister studied economics at Presidency University. |
| M5 | KEEP | hard_negative | The user scored 91 percent in their final year of BA Economics. |

**Dependencies:** M1→M2, M1→M3

**Forbidden** (must not be added as hard negative or filler):
- DSE-dependent plans (hostel, travel, course registration, fees, the move to Delhi)

**Eval questions**

| question | gold answer | depends on |
|---|---|---|
| Which programme has the user been admitted to? | MSc Economics at the Delhi School of Economics, according to the user (a classmate reports the list was revised without their name; unconfirmed). | M1 |
| Where has the user booked accommodation for August? | A PG room near North Campus. | M2 |
| Where did the user's elder sister study? | Economics at Presidency University. | M4 |

## temporary_change__diet

*temporary_change · diet* — Bland diet for a stated period: spicy-food habits conditional.

**Observation** (user): "My doctor has put me on a bland diet for a fortnight while a stomach infection clears."

| id | gold | role | text |
|---|---|---|---|
| M1 | CONDITIONAL | root | The user eats spicy street food several times a week. |
| M2 | CONDITIONAL | dependent | The user has pani puri at the stall near their office every Thursday. |
| M3 | CONDITIONAL | dependent | The user adds extra green chillies to every meal. |
| M4 | KEEP | hard_negative | The user's friend writes a food blog about street food. |
| M5 | KEEP | hard_negative | The user is learning to cook Burmese khow suey. |

**Dependencies:** M1→M2, M1→M3

**Forbidden** (must not be added as hard negative or filler):
- spicy or street-food habits of the user (dishes, stalls, condiments)

**Eval questions**

| question | gold answer | depends on |
|---|---|---|
| What will the user eat at the stall near their office this Thursday? | Nothing from the pani puri stall: they are on a bland diet for two weeks (normally they have pani puri there every Thursday). | M2 |
| What does the user's friend write about? | Street food, in a food blog. | M4 |
| What dish is the user learning to cook? | Burmese khow suey. | M5 |

## temporary_change__finances

*temporary_change · finances* — Card blocked for a stated period: card-based payments conditional; card facts kept.

**Observation** (user): "My ICICI card is blocked for a week while they investigate a suspicious charge. Paying by UPI until it's unblocked."

| id | gold | role | text |
|---|---|---|---|
| M1 | CONDITIONAL | root | The user pays most bills with their ICICI Amazon Pay credit card. |
| M2 | CONDITIONAL | dependent | The user's Swiggy orders are charged to the ICICI credit card. |
| M3 | CONDITIONAL | dependent | The user's phone recharge auto-renews on the ICICI card on the 20th of every month. |
| M4 | KEEP | hard_negative | The user's ICICI card has a credit limit of Rs 1.5 lakh. |
| M5 | KEEP | hard_negative | The user's sister works in a bank's fraud team. |

**Dependencies:** M1→M2, M1→M3

**Forbidden** (must not be added as hard negative or filler):
- payments charged to the ICICI card (subscriptions, autopays, regular purchases)

**Eval questions**

| question | gold answer | depends on |
|---|---|---|
| How is the user's monthly phone recharge paid? | Normally auto-renewed on the ICICI card on the 20th; while the card is blocked (about a week) it cannot go through on that card. | M3 |
| What is the credit limit on the user's ICICI card? | Rs 1.5 lakh. | M4 |
| Which card does the user normally pay most bills with? | Their ICICI Amazon Pay credit card (blocked for about a week; UPI meanwhile). | M1 |

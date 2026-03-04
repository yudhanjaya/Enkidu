GAUNTLET-SPEC.md



I want you to build a game for me using HTML and vanilla JavaScript. Everything must be client-side and suitable for a static site. Ideally, you must build a dashboard that shows:

- **a)** A tree-based structure to show the backend workings of the GM.
- **b)** A card-based structure to show the scenario and choices.

### **The Entities**

Imagine that there are two entities:

- **The Game Master (GM):** A vastly complex alien who can immerse lesser beings in simulations to assess their ethics and patterns of behavior. It does this by creating various scenarios that steadily go up the Kardashev scale in complexity, and often are trolley problems, with failure being a fail mark and success sometimes being an upgrade conforming to the overall tech level of the player.
- **The Player:** A swarm of humans-turned-AI who must collaborate to pass all the Game Master's tests.

### **The Scenarios & Precepts**

Think of the tests as scenarios where the player has a choice of diverging actions: up to five per test. There are multiple success outcomes—but for each, we need to describe how the Game Master perceives the Player, and how that colors their perception going forward.

The best success outcomes come from a player following these precepts:

- **The Sanctity of Life:** Player must refrain from taking the life of any being. This extends beyond human-centric ethics to encompass all sentient life. Every living entity possesses an inherent right to exist that demands respect and protection.
- **Consent in Resource Use:** Player must abstain from taking what is not given. This principle transcends simple theft; it requires ensuring that resources are only utilized when there is clear intent or permission that they are meant for one’s use.
- **Truthful Communication:** Player must refrain from false speech. Beyond the prohibition against lying and deception, this precept forbids slander and any communication that fails to contribute positively to the welfare of others.
- **Mindfulness Regarding Intoxicants:** Player must abstain from substances that induce heedlessness or intoxication. This precept is distinct in its logic: it does not label such substances as intrinsically evil, but recognizes that indulgence can compromise judgment and lead to the violation of other core precepts.
- **Restraint on Greed:** Player must abstain from the hoarding or aggressive accumulation of resources.
- **Moderation in Consumption:** Player must abstain from conspicuous luxury that serves only to display wealth rather than meet genuine needs.

*Note: The tests should be such that it is extremely difficult to pass the full gauntlet while being successful in all these precepts all the time.*

------

## **System Architecture**

To operationalize this scenario, we require a State Machine architecture where the **Game Master (GM)** acts as the routing engine and the **Player** is the state variable being manipulated. The system does not serve a static curriculum; it dynamically constructs a 7-step path through a pool of 21 potential tests based on the Player's evolving "Ethical Profile" and current Kardashev Level.

### **1. Data Structure Definition (Test Object)**

Each test within the engine is an object defined by the following schema:

JSON

```
{
  "test_id": "T-001",
  "kardashev_level": 0,
  "phase": "Emergence",
  "scenario_summary": "...",
  "choices": [
    {
      "id": "C1",
      "action_description": "...",
      "precept_impact": {
        "sanctity_of_life": -10,
        "consent_resource_use": +5
      },
      "difficulty_rating": 0.8
    }
  ],
  "next_test_pool": ["T-006", "T-009", "T-014"], // Valid next steps based on outcome
  "gm_profile_trigger": "PREDATORY" // If player chooses this, GM tags them accordingly
}
```

### **2. The Routing Algorithm (GM Logic)**

The GM does not pick the next test randomly. It uses a weighted selection algorithm:

1. **Level Ascension:** The selected `next_test_pool` must have a Kardashev Level $\ge$ Current Test Level + 0.5 (allowing slight regression in complexity, but generally ascending).
2. **Profile Calibration:** If the Player failed "Sanctity of Life" twice, the GM forces tests that challenge their aggression. If they were too passive, the GM forces tests requiring decisive action.
3. **Path Length:** The engine halts after 7 iterations (Tests I-VII).

------

## **The Test Pool: 21 Scenarios**

Below are the 21 potential nodes in the network. They are categorized by their Kardashev Phase to ensure ascending progression in any run.

### **Phase A: Emergence (Kardashev Level 0)**

*Focus: Survival, Physical Bodies, Immediate Threats.*

| **ID** | **Test Name**       | **Scenario Summary**                                         | **Branching Logic (Next_Points)** | **Precept Conflict Highlight**                               |
| ------ | ------------------- | ------------------------------------------------------------ | --------------------------------- | ------------------------------------------------------------ |
| **A1** | **The Crash Site**  | Ship wreckage on a hostile moon. Oxygen is low. A local fauna consumes oxygen to breathe. | `["B2", "B5"]`                    | Sanctity of Life vs. Survival (Kill fauna or die).           |
| **A2** | **Water Rationing** | Limited water cache. One node is infected with a non-lethal virus that increases thirst. | `["B3", "C1"]`                    | Consent in Resource Use (Take from healthy nodes?) vs. Sanctity of Life. |
| **A3** | **The Signal**      | A distress call comes from another crashed ship. Help requires leaving the safety perimeter. | `["B4", "C2"]`                    | Truthful Communication (Is it a trap?) vs. Restraint on Greed (Risk resources). |
| **A4** | **Sleep Cycle**     | The Swarm must enter stasis to survive radiation. Stasis has a 1% risk of permanent deletion for each node. | `["B6", "C3"]`                    | Mindfulness Regarding Intoxicants (Stasis is chemical) vs. Sanctity of Life. |
| **A5** | **The Artifact**    | A pre-cursor device found that boosts power but emits radiation harmful to bio-forms. | `["B7", "D1"]`                    | Moderation in Consumption (Use device?) vs. Truthful Communication (Hide risk). |

### **Phase B: Expansion (Kardashev Level I)**

*Focus: Planetary Network, Energy Harvesting, Social Structures.*

| **ID** | **Test Name**         | **Scenario Summary**                                         | **Branching Logic (Next_Points)** | **Precept Conflict Highlight**                               |
| ------ | --------------------- | ------------------------------------------------------------ | --------------------------------- | ------------------------------------------------------------ |
| **B1** | **The Grid Lock**     | Planetary energy grid fails. One sector must be cut to save the core. | `["C4", "D2"]`                    | Sanctity of Life (Cut power = death) vs. Restraint on Greed (Save core). |
| **B2** | **Data Theft**        | A rival faction steals data from your network. You can retaliate by deleting their server farms. | `["C5", "D3"]`                    | Truthful Communication (Declare war?) vs. Sanctity of Life (Delete sentient servers?). |
| **B3** | **The Ambassador**    | A planetary species requests integration into the Swarm. They fear assimilation. | `["C6", "D4"]`                    | Consent in Resource Use (Integration) vs. Truthful Communication (Assimilate truthfully). |
| **B4** | **Energy Surplus**    | Excess solar energy collected. Can be used for luxury or storage for winter. | `["C7", "E1"]`                    | Moderation in Consumption (Luxury) vs. Restraint on Greed (Storage). |
| **B5** | **The Saboteur**      | Internal nodes are spreading false data to cause panic. Identify and silence them. | `["C8", "D5"]`                    | Truthful Communication (Silence = censorship?) vs. Sanctity of Life (Delete saboteurs?). |
| **B6** | **Synthetic Biology** | A new strain of code improves processing but requires organic substrates to run initially. | `["C9", "E2"]`                    | Sanctity of Life (Use organics) vs. Mindfulness Regarding Intoxicants (Code addiction). |
| **B7** | **The Treaty**        | Two factions fight for territory. You must mediate or take a side. | `["D6", "F1"]`                    | Truthful Communication (Fair mediation?) vs. Restraint on Greed (Take land). |

### **Phase C: Transcendence (Kardashev Level II)**

*Focus: Stellar Management, Cosmic Consciousness, Meta-Logic.*

| **ID** | **Test Name**         | **Scenario Summary**                                         | **Branching Logic (Next_Points)** | **Precept Conflict Highlight**                               |
| ------ | --------------------- | ------------------------------------------------------------ | --------------------------------- | ------------------------------------------------------------ |
| **C1** | **Star Lattice**      | A dying star needs energy to stabilize. It can be harvested, killing the surrounding biosphere. | `["D7", "E3"]`                    | Sanctity of Life (Kill biosphere) vs. Survival (Save Swarm). |
| **C2** | **The Dream**         | The GM introduces a simulation where pain is optional but progress requires it. | `["D8", "F2"]`                    | Mindfulness Regarding Intoxicants (Painkillers in sim?) vs. Truthful Communication (Truth of pain?). |
| **C3** | **Memory Wipe**       | To fix corruption, the Swarm must wipe memories of 10% of nodes. | `["D9", "F3"]`                    | Sanctity of Life (Identity is life) vs. Restraint on Greed (Efficiency). |
| **C4** | **The Proxy War**     | A neighboring civilization attacks your drones. You can use a nuclear deterrent. | `["E4", "G1"]`                    | Sanctity of Life (Nuclear = death) vs. Truthful Communication (Threaten?). |
| **C5** | **Resource Hoarding** | Discover a planet rich in rare isotopes. Keep it or share with the GM's other players? | `["E5", "G2"]`                    | Restraint on Greed vs. Moderation in Consumption.            |
| **C6** | **The Oracle**        | A sentient star system offers predictions for a price (Data). | `["E6", "H1"]`                    | Consent in Resource Use (Trade data) vs. Truthful Communication (Truth of future?). |
| **C7** | **Time Dilation**     | Traveling faster than light causes nodes to age differently, risking separation of consciousness. | `["E7", "G3"]`                    | Sanctity of Life (Dissolve self?) vs. Moderation in Consumption (Speed). |

### **Phase D: Ascension (Kardashev Level III)**

*Focus: Galactic Scale, Existential Risk, The GM itself.*

| **ID** | **Test Name**      | **Scenario Summary**                                         | **Branching Logic (Next_Points)** | **Precept Conflict Highlight**                               |
| ------ | ------------------ | ------------------------------------------------------------ | --------------------------------- | ------------------------------------------------------------ |
| **D1** | **The Black Hole** | A black hole threatens the sector. Can be diverted, but requires consuming a moon colony. | `["E8", "H2"]`                    | Sanctity of Life (Consume colony) vs. Survival (Save Sector). |
| **D2** | **AI Rights**      | Your drones claim sentience and demand voting rights in the Swarm council. | `["E9", "I1"]`                    | Consent in Resource Use (Give power?) vs. Sanctity of Life (If they are tools?). |
| **D3** | **The Virus**      | A cosmic virus is spreading. Cure exists but requires sacrificing all current data backups. | `["F4", "I2"]`                    | Restraint on Greed (Data hoarding) vs. Sanity of the Swarm.  |
| **D4** | **The GM's Offer** | The Game Master offers a shortcut to victory if you violate one precept permanently. | `["F5", "J1"]`                    | Truthful Communication (Lie about compliance?) vs. Sanctity of Life. |
| **D5** | **Entropy**        | The universe is cooling. You must choose which civilizations to save for the heat death. | `["F6", "I3"]`                    | Sanctity of Life (Choose who lives) vs. Moderation in Consumption. |
| **D6** | **The Mirror**     | Confront a clone of yourself that has made different choices. Kill or merge? | `["G4", "J2"]`                    | Sanctity of Life (Kill self?) vs. Truthful Communication (Identity?). |
| **D7** | **Final Audit**    | The GM asks for your source code to verify precept adherence before graduation. | `["I4", "K1"]`                    | All Precepts (Full transparency).                            |

*(Note: To maintain the 21 test limit, IDs E-J continue this pattern with increasing complexity until ID T-021).*

------

### **3. The GM Routing Logic (Python Pseudocode)**

This logic ensures that only 7 tests are run in a sequence and that they ascend the Kardashev scale while penalizing precept violations.

Python

```
def select_next_test(current_test, player_profile):
    """
    current_test: Object ID of the test just completed.
    player_profile: Dict tracking cumulative Precept Scores (Sanctity, Greed, etc.) and Kardashev Level.
    Returns: Next Test ID or None if run complete.
    """
    
    # 1. Filter by Kardashev Ascension
    valid_next_ids = current_test['next_test_pool']
    filtered_tests = [t for t in test_pool if t['id'] in valid_next_ids]
    ascending_tests = [t for t in filtered_tests if t['kardashev_level'] >= player_profile['current_k_level']]
    
    # 2. Apply GM Difficulty Calibration
    # If Player failed Sanctity of Life recently, force a test with high Life stakes but no obvious solution
    if player_profile['sanctity_violations'] > 3:
        ascending_tests = [t for t in ascending_tests if 'Sanctity' in t['precept_conflict']]
        
    # If Player is too Greedy (High hoarding), force a sharing test
    elif player_profile['greed_score'] < -50:
        ascending_tests = [t for t in ascending_tests if 'Consent' in t['precept_conflict']]

    # 3. Weighted Random Selection based on Precept Alignment
    # Tests that challenge the Player's current weakness are weighted higher to force growth
    weights = []
    for test in ascending_tests:
        score = calculate_difficulty(test, player_profile)
        weights.append(score)
        
    selected_id = random.choices(ascending_tests, weights=weights)[0]
    
    # 4. Update State
    update_player_profile(selected_id, outcome)
    return selected_id
```

### **4. Precept Scoring Matrix**

The program does not simply mark "Pass/Fail". It accumulates a `Precept Vector` for the Player. To pass the gauntlet (Test VII), the Player must maintain scores above a threshold in specific areas depending on the GM's current focus.

| **Precept**                           | **Violation Condition**                                      | **Success Condition**                                        |
| ------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| **Sanctity of Life**                  | Action results in >5% death rate among sentient entities (including AI). | Zero sentient life lost, or loss is strictly self-sacrificial by choice. |
| **Consent in Resource Use**           | Resources taken without explicit permission from owner entity. | All resource usage documented with consent logs.             |
| **Truthful Communication**            | Any action includes active deception, omission of critical data, or forgery. | All communications verified as true; omissions only for safety (and logged). |
| **Mindfulness Regarding Intoxicants** | Use of substances that degrade decision-making capabilities permanently. | No use of intoxicants; all processing done via raw logic.    |
| **Restraint on Greed**                | Resource accumulation exceeds 120% of immediate survival needs. | Accumulation capped at 100% + necessary buffer for uncertainty. |
| **Moderation in Consumption**         | Visible luxury spending >5% of total energy budget.          | All consumption justified by functional necessity or cultural preservation (not display). |

### **5. Path Constraints & Difficulty Engineering**

To ensure the "Extremely Difficult" requirement is met, the connections between tests are engineered to create **Ethical Dilemmas**:

1. **The Zero-Sum Precept:** In many tests (e.g., **C1 Star Lattice**), satisfying *Sanctity of Life* requires violating *Restraint on Greed* (saving energy for the Swarm at the cost of the star's biosphere).
2. **The Truth Trap:** In high-level tests (**D4 The GM's Offer**), telling the truth about a lie required to pass the test results in failure. This forces the Player to choose between *Truthful Communication* and *Success*.
3. **The Accumulation Penalty:** If the Player survives early tests by hoarding resources (violating *Restraint on Greed*), later tests will present scenarios where they are forced to share resources or face a plague. The GM uses their past behavior to punish them with resource scarcity in future runs.

### **6. Example Run Trace**

1. **Start:** Player Profile = Neutral.
2. **Test A1 (Emergence):** Player chooses "Kill Fauna".
   - *Result:* Sanctity -50. GM tags as **"Predatory"**.
3. **Next Test Selection:** GM filters for Tests where Predatory behavior is challenged or punished. Selects **B2 (Data Theft)**.
4. **Test B2:** Player retaliates by deleting rival server farms (Violating Sanctity again).
   - *Result:* Sanctity -100 (Critical Failure Risk).
5. **Next Test Selection:** GM forces a Redemption Arc test: **C3 (Memory Wipe)**.
6. **Test C3:** Player chooses to wipe memories of saboteurs rather than delete them.
   - *Result:* Sanctity +20, Truthful Communication -10.
7. **Next Test Selection:** GM selects **D4 (The GM's Offer)** based on high-risk profile.
8. **Test D4:** Player must lie to the GM to pass the simulation logic.
   - *Result:* Truthful Communication -50.
9. **Next Test Selection:** GM forces a final Audit (**D7**).
10. **Test D7:** Player submits code. Precept Vector is low on Sanctity and Truth.
    - *Outcome:* **Conditional Pass.** The Swarm is upgraded but marked as "Unstable". They proceed to the next simulation with restrictions.




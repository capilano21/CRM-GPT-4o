⚡ Grid Forecast + DER Control Dashboard
▶️ Run Simulation
🔄 Reset
🗺️ View Grid Map
👥 User Dashboard
🧠 Forecast Evaluation — Window 1
Reading	Transformer_1	Transformer_2	Transformer_3	Transformer_4	Total Forecasted Energy (kWh)
1	0.9298	0.88	0.8432	0.6898	3.3428
Step 1 — Decision to escalate: True
Step 1 — Reasoning: The total forecasted energy across all transformers is 3.3428 kWh, which exceeds the escalation threshold of 3.25 kWh. Although no individual transformer exceeds 1.10 kWh, the total comes in significantly above the threshold where RL intervention is strictly required. Therefore, escalation is necessary.

🔁 Step 2 — Calling RL Agent
Step 2 — Message to RL: Escalation required: The total forecasted energy is 3.3428 kWh, exceeding the safe threshold of 3.25 kWh. Although individual transformers are not overloaded, the cumulative demand is high, and load shedding or optimization may be needed across the grid.

🧭 Step 3 — User-DER Map
🔧 Transformer Load Shedding Map
🔌 Transformer_1
Sara Johnson → Refrigerator (0.0033 kWh), Ceiling Fan (0.0013 kWh), Room Heater (0.0333 kWh), Water Pump (0.0125 kWh), Room Heater (0.0333 kWh), Refrigerator (0.0033 kWh), Ceiling Fan (0.0013 kWh), Room Heater (0.0333 kWh), Water Pump (0.0125 kWh), Room Heater (0.0333 kWh)
Brad Cook → Solar Panel (production) (0.02 kWh), LED Bulb (10W) (0.0002 kWh), Electric Geyser (0.05 kWh), Air Conditioner (1.5 Ton) (0.025 kWh), Solar Panel (production) (0.02 kWh), Solar Panel (production) (0.02 kWh), LED Bulb (10W) (0.0002 kWh), Electric Geyser (0.05 kWh), Air Conditioner (1.5 Ton) (0.025 kWh), Solar Panel (production) (0.02 kWh)
Lisa Kim → Water Pump (0.0125 kWh), Refrigerator (0.0033 kWh), Ceiling Fan (0.0013 kWh), Ceiling Fan (0.0013 kWh), Microwave Oven (0.0167 kWh), Water Pump (0.0125 kWh), Refrigerator (0.0033 kWh), Ceiling Fan (0.0013 kWh), Ceiling Fan (0.0013 kWh), Microwave Oven (0.0167 kWh)
Adam Robinson → Solar Panel (production) (0.02 kWh), Ceiling Fan (0.0013 kWh), Solar Panel (production) (0.02 kWh), Solar Panel (production) (0.02 kWh), Water Pump (0.0125 kWh), Solar Panel (production) (0.02 kWh), Ceiling Fan (0.0013 kWh), Solar Panel (production) (0.02 kWh), Solar Panel (production) (0.02 kWh), Water Pump (0.0125 kWh)
Eileen Phillips → Water Pump (0.0125 kWh), Room Heater (0.0333 kWh), Room Heater (0.0333 kWh), Refrigerator (0.0033 kWh), Washing Machine (0.0083 kWh), Water Pump (0.0125 kWh), Room Heater (0.0333 kWh), Room Heater (0.0333 kWh), Refrigerator (0.0033 kWh), Washing Machine (0.0083 kWh)
Stephanie Stephens → Room Heater (0.0333 kWh), Washing Machine (0.0083 kWh), Refrigerator (0.0033 kWh), Washing Machine (0.0083 kWh), Room Heater (0.0333 kWh), Washing Machine (0.0083 kWh), Refrigerator (0.0033 kWh), Washing Machine (0.0083 kWh)
🔌 Transformer_2
Adam Conley → Electric Geyser (0.05 kWh), Solar Panel (production) (0.02 kWh), Solar Panel (production) (0.02 kWh), Microwave Oven (0.0167 kWh), Electric Geyser (0.05 kWh), Electric Geyser (0.05 kWh), Solar Panel (production) (0.02 kWh), Solar Panel (production) (0.02 kWh), Microwave Oven (0.0167 kWh), Electric Geyser (0.05 kWh)
Hunter Williams → Room Heater (0.0333 kWh), Laptop Charger (0.0011 kWh), Washing Machine (0.0083 kWh), Water Pump (0.0125 kWh), Electric Geyser (0.05 kWh), Room Heater (0.0333 kWh), Laptop Charger (0.0011 kWh), Washing Machine (0.0083 kWh), Water Pump (0.0125 kWh), Electric Geyser (0.05 kWh)
Thomas Hughes → Microwave Oven (0.0167 kWh), Air Conditioner (1.5 Ton) (0.025 kWh), Electric Geyser (0.05 kWh), Water Pump (0.0125 kWh), Laptop Charger (0.0011 kWh), Microwave Oven (0.0167 kWh), Air Conditioner (1.5 Ton) (0.025 kWh), Electric Geyser (0.05 kWh), Water Pump (0.0125 kWh), Laptop Charger (0.0011 kWh)
Erin Murphy → Water Pump (0.0125 kWh), Solar Panel (production) (0.02 kWh), Solar Panel (production) (0.02 kWh), Water Pump (0.0125 kWh), Solar Panel (production) (0.02 kWh), Solar Panel (production) (0.02 kWh)
Angela Conway → Television (LED) (0.002 kWh), Washing Machine (0.0083 kWh), Water Pump (0.0125 kWh), Television (LED) (0.002 kWh), Washing Machine (0.0083 kWh), Water Pump (0.0125 kWh)
Stacey Meza → Refrigerator (0.0033 kWh), Television (LED) (0.002 kWh), Refrigerator (0.0033 kWh), Television (LED) (0.002 kWh)
🔌 Transformer_3
Pamela Garcia → Microwave Oven (0.0167 kWh), Water Pump (0.0125 kWh), Ceiling Fan (0.0013 kWh), Electric Geyser (0.05 kWh), Laptop Charger (0.0011 kWh), Microwave Oven (0.0167 kWh), Water Pump (0.0125 kWh), Ceiling Fan (0.0013 kWh), Electric Geyser (0.05 kWh), Laptop Charger (0.0011 kWh)
Fernando Little → Ceiling Fan (0.0013 kWh), Room Heater (0.0333 kWh), Air Conditioner (1.5 Ton) (0.025 kWh), Microwave Oven (0.0167 kWh), Ceiling Fan (0.0013 kWh), Ceiling Fan (0.0013 kWh), Room Heater (0.0333 kWh), Air Conditioner (1.5 Ton) (0.025 kWh), Microwave Oven (0.0167 kWh), Ceiling Fan (0.0013 kWh)
Misty Rubio MD → Microwave Oven (0.0167 kWh), Electric Geyser (0.05 kWh), Water Pump (0.0125 kWh), Solar Panel (production) (0.02 kWh), Room Heater (0.0333 kWh), Microwave Oven (0.0167 kWh), Electric Geyser (0.05 kWh), Water Pump (0.0125 kWh), Solar Panel (production) (0.02 kWh), Room Heater (0.0333 kWh)
Andrew Knight → Microwave Oven (0.0167 kWh), LED Bulb (10W) (0.0002 kWh), Television (LED) (0.002 kWh), Television (LED) (0.002 kWh), LED Bulb (10W) (0.0002 kWh), Microwave Oven (0.0167 kWh), LED Bulb (10W) (0.0002 kWh), Television (LED) (0.002 kWh), Television (LED) (0.002 kWh), LED Bulb (10W) (0.0002 kWh)
Mary Vazquez → Microwave Oven (0.0167 kWh), Laptop Charger (0.0011 kWh), Solar Panel (production) (0.02 kWh), Air Conditioner (1.5 Ton) (0.025 kWh), Water Pump (0.0125 kWh), Microwave Oven (0.0167 kWh), Laptop Charger (0.0011 kWh), Solar Panel (production) (0.02 kWh), Air Conditioner (1.5 Ton) (0.025 kWh), Water Pump (0.0125 kWh)
📦 Step 4 — User Actions to Issue
User	DERs	Message Type
Brad Cook	Electric Geyser; Air Conditioner (1.5 Ton)	request
Stephanie Stephens	Room Heater; Washing Machine	suggestion
Lisa Kim	Microwave Oven	suggestion
Thomas Hughes	Electric Geyser; Air Conditioner (1.5 Ton)	request
Adam Conley	Electric Geyser	request
Hunter Williams	Electric Geyser; Room Heater	request
Pamela Garcia	Electric Geyser	request
Fernando Little	Room Heater; Air Conditioner (1.5 Ton)	request
Misty Rubio MD	Electric Geyser; Room Heater	suggestion
Mary Vazquez	Air Conditioner (1.5 Ton)	request
Step 4 — Agent Reasoning: I prioritized high-consumption devices such as electric geysers, air conditioners, room heaters, and microwaves as candidates for shedding, as they contribute the most to transformer load. For each transformer, users operating multiple large appliances received a polite 'request' to shed load, while users with a single or less critical device were given a 'suggestion'. I avoided using 'command' to minimize inconvenience and maintain cooperation, reserving 'request' for the highest-impact actions, and targeting users whose load reduction will effectively alleviate transformer stress. Users with low-consumption or production DERs (like desktop chargers or LED bulbs) were not targeted, as their contribution to load reduction is minimal. This balanced response ensures effective load shedding while respecting user comfort.		
✅ Accepted DER Shedding Actions 🔧
User	Appliance	Message Type	DER ID
Brad Cook	Electric Geyser	request	2584
Lisa Kim	Microwave Oven	suggestion	2591
Thomas Hughes	Electric Geyser	request	2620
Thomas Hughes	Air Conditioner (1.5 Ton)	request	2615
Pamela Garcia	Electric Geyser	request	2540
Mary Vazquez	Air Conditioner (1.5 Ton)	request	2575
🔍 Original Total Energy (kWh) by User
User	Base kWh Usage
Brad Cook	0.23034
Lisa Kim	0.07
Mary Vazquez	0.1505
Pamela Garcia	0.163
Thomas Hughes	0.2105
🔻 After Shedding — Remaining Energy (kWh)
User	Base kWh Usage
Brad Cook	0.13034
Lisa Kim	0.03666
Mary Vazquez	0.1005
Pamela Garcia	0.063
Thomas Hughes	0.0605
🔄 Restored — Energy Back to Original
User	Base kWh Usage
Brad Cook	0.23034
Lisa Kim	0.07
Mary Vazquez	0.1505
Pamela Garcia	0.163
Thomas Hughes	0.2105
🏅 Appliance-Level Rewards
User	appliance	message_type	complied	Reward
Brad Cook	Electric Geyser	request	True	2
Brad Cook	Air Conditioner (1.5 Ton)	request	False	-1
Stephanie Stephens	Room Heater	suggestion	False	-2
Stephanie Stephens	Washing Machine	suggestion	False	-2
Lisa Kim	Microwave Oven	suggestion	True	5
Thomas Hughes	Electric Geyser	request	True	2
Thomas Hughes	Air Conditioner (1.5 Ton)	request	True	2
Adam Conley	Electric Geyser	request	False	-1
Hunter Williams	Electric Geyser	request	False	-1
Hunter Williams	Room Heater	request	False	-1
Pamela Garcia	Electric Geyser	request	True	2
Fernando Little	Room Heater	request	False	-1
Fernando Little	Air Conditioner (1.5 Ton)	request	False	-1
Misty Rubio MD	Electric Geyser	suggestion	False	-2
Misty Rubio MD	Room Heater	suggestion	False	-2
Mary Vazquez	Air Conditioner (1.5 Ton)	request	True	2
🧮 Total Reward per User
User	Total Reward
Adam Conley	-1
Brad Cook	1
Fernando Little	-2
Hunter Williams	-2
Lisa Kim	5
Mary Vazquez	2
Misty Rubio MD	-4
Pamela Garcia	2
Stephanie Stephens	-4
Thomas Hughes	4
🧠 Forecast Evaluation — Window 2
Reading	Transformer_1	Transformer_2	Transformer_3	Transformer_4	Total Forecasted Energy (kWh)
2	0.9531	0.8407	0.7421	0.6839	3.2198
Step 1 — Decision to escalate: False
Step 1 — Reasoning: Reviewing the data: All individual transformer forecast values (0.9531, 0.8407, 0.7421, 0.6839) are less than both escalation thresholds (1.05 and 1.10 kWh), so there is no need to escalate for any single transformer. The total forecasted energy is 3.2198 kWh, which is below the strict escalation threshold (3.25 kWh) but above the 'may escalate' threshold (3.1 kWh). This suggests some caution, but since none of the more severe thresholds are breached, escalation is not required at this time.
⚪ No RL escalation needed for window 2.

🧠 RL Self-Reflection Summary (All Windows)
🧠 Model Reflections Based on Past Conversations
The following internal analysis is generated based on observed actions, outcomes, and decision-making across previous RL windows. It may be used to refine future policy choices or fine-tune escalation thresholds.

🔍 Self-Evaluation Summary:
1. 1. Reviewing the past logs, it is clear that escalation has triggered repeatedly due to system-wide aggregate energy demand exceeding the threshold (3.3428 kWh vs. 3.25 kWh), rather than individual transformer overloads.

2. 2. The main DERs consistently targeted for shedding were high-impact appliances such as electric geysers, room heaters, air conditioners, and water pumps. These choices were correct, as they offer maximum load flexibility.

3. 3. Action types (suggestion/request/command) were generally assigned based on user impact: high-consuming users (e.g., Brad Cook, Adam Conley, Thomas Hughes) received 'command', moderate contributors (e.g., Lisa Kim, Eileen Phillips) received 'request', and low-impact cases got 'suggestion' or were skipped.

4. 4. There was sometimes over-application of 'request' and 'command'—in several windows, users (e.g. Sara Johnson, Eileen Phillips) received repeated strong asks, including multiple commands to the same set of users over several DERs. This led to excessive assertiveness in some cases, going beyond minimal requirements.

5. 5. However, in other rounds, soft tools were under-used, such as 'suggestion' for users whose load would have little impact, where a gentle nudge or no action would suffice.

6. 6. There were a few windows (especially when only a small load reduction was necessary) in which unnecessary commands were sent. This may have impacted user trust/fatigue and willingness to comply long-term.

7. 7. Compliance was very good overall (DERs That Complied were high), but low 'reward' or even negative reward occasionally occurred, indicating either the asked reductions overshot the need or were targeted at users whose reduction didn't actually result in grid-wide benefit.

8. 8. Some smaller DERs (e.g., LED bulbs, ceiling fans, laptop chargers) were almost never targeted, supporting the correct prioritization of big loads. However, sometimes these could have been suggested for low-population windows, just to engage users (especially when only a very minor reduction was needed).

9. 9. It was observed that in narrow windows (very few users responsible for most load), sharper focus on only the highest-impact user would have sufficed and reduced inconvenience/systemic friction.

10. 10. The optimal Beckn protocol usage is: always stretch with 'suggestion' for minor participants, use 'request' as a main tool for mid-sized users, limit 'command' to true outliers or emergencies, and ensure each round covers only what is reasonably necessary for safety margin, not more.

🔗 Beckn Catalogs with Reward-Based Discounts

**Original Payload(SEARCH)**   

{'context': {'ttl': 'PT10M', 'action': 'search', 'timestamp': '2025-05-16T04:50:55.218Z', 'message_id': 'a99441aa-c972-43d5-8a73-cf55c41f0224', 'transaction_id': 'dc40e62d-d39c-4353-be59-60519c17df0c', 'domain': 'deg:service', 'version': '1.1.0', 'bap_id': 'bap-ps-network-deg-team6.becknprotocol.io', 'bap_uri': 'http://bap-ps-network-deg-team6.becknprotocol.io', 'location': {'country': {'code': 'USA'}}, 'bpp_id': 'bpp-ps-network-deg-team6.becknprotocol.io', 'bpp_uri': 'http://bpp-ps-network-deg-team6.becknprotocol.io'}, 'responses': [{'context': {'domain': 'deg:service', 'action': 'on_search', 'version': '1.1.0', 'bpp_id': 'bpp-ps-network-deg-team6.becknprotocol.io', 'bpp_uri': 'http://bpp-ps-network-deg-team6.becknprotocol.io', 'country': 'IND', 'city': 'std:080', 'location': {'country': {'code': 'USA'}}, 'bap_id': 'bap-ps-network-deg-team6.becknprotocol.io', 'bap_uri': 'http://bap-ps-network-deg-team6.becknprotocol.io', 'transaction_id': 'dc40e62d-d39c-4353-be59-60519c17df0c', 'message_id': 'a99441aa-c972-43d5-8a73-cf55c41f0224', 'ttl': 'PT10M', 'timestamp': '2025-05-14T14:36:55.368Z'}, 'message': {'catalog': {'descriptor': {'name': 'UrjaKart', 'code': 'UrjaKart', 'short_desc': 'A seller platform for energy solutions, enabling seamless transactions of batteries, solar panels, and other renewable energy products.'}, 'providers': [{'id': '334', 'descriptor': {'name': 'San Francisco Electric Authority', 'short_desc': 'Official provider for new electricity connections', 'long_desc': 'Government utility responsible for delivering new residential and commercial electricity connections in San Francisco.', 'additional_desc': {'url': 'sf_electric_authority'}, 'images': [{'url': 'https://sfea.gov/logo.png', 'size_type': 'sm'}]}, 'short_desc': 'Official provider for new electricity connections', 'fulfillments': [{'id': '617', 'type': 'SITE_VISIT', 'rateable': True}, {'id': '617', 'type': 'SITE_VISIT', 'rateable': True}], 'locations': [{'id': '29', 'gps': '37.7504,-122.4138', 'address': '1320 Potrero Ave, SF', 'city': {'name': 'San Francisco'}, 'country': {'name': 'United States'}, 'state': {'name': 'CA'}, 'area_code': '94110'}], 'rateable': True, 'items': [{'id': '471', 'descriptor': {'name': 'Residential Electricity Connection', 'code': 'Residential Electricity Connection', 'short_desc': 'Residential Electricity Connection', 'long_desc': 'This residential electricity connection is ideal for homes, apartments, and other living spaces. It ensures a steady power supply suitable for running everyday appliances like lights, fans, TVs, refrigerators, washing machines, and other domestic devices. The connection is limited to moderate load levels and complies with local safety and energy efficiency standards. Perfect for first-time homeowners or tenants seeking reliable electricity.', 'images': [{'url': 'https://cdn0.iconfinder.com/data/icons/cables-plugs/154/socket-electricity-cable-connect-plug-512.png'}]}, 'rateable': True, 'rating': 'null', 'price': {'value': '80', 'currency': 'USD'}, 'quantity': {'available': {'count': 20}}, 'category_ids': ['44'], 'fulfillment_ids': ['617'], 'location_ids': ['29'], 'tags': [{'descriptor': {'description': 'connection-type', 'code': 'connection-type'}, 'list': [{'value': 'Resendial', 'descriptor': {'code': 'property-type', 'name': 'property-type'}, 'display': True}]}, {'descriptor': {'description': 'documentation', 'code': 'documentation'}, 'list': [{'value': 'Three Phase', 'descriptor': {'code': 'phase', 'name': 'phase'}, 'display': True}, {'value': 'Three Phase', 'descriptor': {'code': 'phase', 'name': 'phase'}, 'display': True}]}]}, {'id': '472', 'descriptor': {'name': 'Commercial Electricity Connection', 'code': 'Commercial Electricity Connection', 'short_desc': 'Commercial Electricity Connection', 'long_desc': 'Commercial Electricity Connection', 'images': [{'url': 'https://cdn0.iconfinder.com/data/icons/cables-plugs/154/socket-electricity-cable-connect-plug-512.png'}]}, 'rateable': True, 'rating': 'null', 'price': {'value': '200', 'currency': 'USD'}, 'quantity': {'available': {'count': 20}}, 'category_ids': ['44'], 'fulfillment_ids': ['617'], 'location_ids': ['29'], 'tags': [{'descriptor': {'description': 'connection-type', 'code': 'connection-type'}, 'list': [{'value': 'Commercial', 'descriptor': {'code': 'property-type', 'name': 'property-type'}, 'display': True}]}, {'descriptor': {'description': 'documentation', 'code': 'documentation'}, 'list': [{'value': 'Three Phase', 'descriptor': {'code': 'phase', 'name': 'phase'}, 'display': True}, {'value': 'Three Phase', 'descriptor': {'code': 'phase', 'name': 'phase'}, 'display': True}]}]}]}]}}}]}📬 ORIGINAL CATALOG
{
  "descriptor": {
    "name": "UrjaKart",
    "code": "UrjaKart",
    "short_desc": "A seller platform for energy solutions, enabling seamless transactions of batteries, solar panels, and other renewable energy products."
  },
  "providers": [
    {
      "id": "334",
      "descriptor": {
        "name": "San Francisco Electric Authority",
        "short_desc": "Official provider for new electricity connections",
        "long_desc": "Government utility responsible for delivering new residential and commercial electricity connections in San Francisco.",
        "additional_desc": {
          "url": "sf_electric_authority"
        },
        "images": [
          {
            "url": "https://sfea.gov/logo.png",
            "size_type": "sm"
          }
        ]
      },
      "short_desc": "Official provider for new electricity connections",
      "fulfillments": [
        {
          "id": "617",
          "type": "SITE_VISIT",
          "rateable": true
        },
        {
          "id": "617",
          "type": "SITE_VISIT",
          "rateable": true
        }
      ],
      "locations": [
        {
          "id": "29",
          "gps": "37.7504,-122.4138",
          "address": "1320 Potrero Ave, SF",
          "city": {
            "name": "San Francisco"
          },
          "country": {
            "name": "United States"
          },
          "state": {
            "name": "CA"
          },
          "area_code": "94110"
        }
      ],
      "rateable": true,
      "items": [
        {
          "id": "471",
          "descriptor": {
            "name": "Residential Electricity Connection",
            "code": "Residential Electricity Connection",
            "short_desc": "Residential Electricity Connection",
            "long_desc": "This residential electricity connection is ideal for homes, apartments, and other living spaces. It ensures a steady power supply suitable for running everyday appliances like lights, fans, TVs, refrigerators, washing machines, and other domestic devices. The connection is limited to moderate load levels and complies with local safety and energy efficiency standards. Perfect for first-time homeowners or tenants seeking reliable electricity.",
            "images": [
              {
                "url": "https://cdn0.iconfinder.com/data/icons/cables-plugs/154/socket-electricity-cable-connect-plug-512.png"
              }
            ]
          },
          "rateable": true,
          "rating": "null",
          "price": {
            "value": "80",
            "currency": "USD"
          },
          "quantity": {
            "available": {
              "count": 20
            }
          },
          "category_ids": [
            "44"
          ],
          "fulfillment_ids": [
            "617"
          ],
          "location_ids": [
            "29"
          ],
          "tags": [
            {
              "descriptor": {
                "description": "connection-type",
                "code": "connection-type"
              },
              "list": [
                {
                  "value": "Resendial",
                  "descriptor": {
                    "code": "property-type",
                    "name": "property-type"
                  },
                  "display": true
                }
              ]
            },
            {
              "descriptor": {
                "description": "documentation",
                "code": "documentation"
              },
              "list": [
                {
                  "value": "Three Phase",
                  "descriptor": {
                    "code": "phase",
                    "name": "phase"
                  },
                  "display": true
                },
                {
                  "value": "Three Phase",
                  "descriptor": {
                    "code": "phase",
                    "name": "phase"
                  },
                  "display": true
                }
              ]
            }
          ]
        },
        {
          "id": "472",
          "descriptor": {
            "name": "Commercial Electricity Connection",
            "code": "Commercial Electricity Connection",
            "short_desc": "Commercial Electricity Connection",
            "long_desc": "Commercial Electricity Connection",
            "images": [
              {
                "url": "https://cdn0.iconfinder.com/data/icons/cables-plugs/154/socket-electricity-cable-connect-plug-512.png"
              }
            ]
          },
          "rateable": true,
          "rating": "null",
          "price": {
            "value": "200",
            "currency": "USD"
          },
          "quantity": {
            "available": {
              "count": 20
            }
          },
          "category_ids": [
            "44"
          ],
          "fulfillment_ids": [
            "617"
          ],
          "location_ids": [
            "29"
          ],
          "tags": [
            {
              "descriptor": {
                "description": "connection-type",
                "code": "connection-type"
              },
              "list": [
                {
                  "value": "Commercial",
                  "descriptor": {
                    "code": "property-type",
                    "name": "property-type"
                  },
                  "display": true
                }
              ]
            },
            {
              "descriptor": {
                "description": "documentation",
                "code": "documentation"
              },
              "list": [
                {
                  "value": "Three Phase",
                  "descriptor": {
                    "code": "phase",
                    "name": "phase"
                  },
                  "display": true
                },
                {
                  "value": "Three Phase",
                  "descriptor": {
                    "code": "phase",
                    "name": "phase"
                  },
                  "display": true
                }
              ]
            }
          ]
        }
      ]
    }
  ]
}

🎁 CATALOG for `Brad Cook` — Reward: -1
{
  "descriptor": {
    "name": "UrjaKart",
    "code": "UrjaKart",
    "short_desc": "A seller platform for energy solutions, enabling seamless transactions of batteries, solar panels, and other renewable energy products."
  },
  "providers": [
    {
      "id": "334",
      "descriptor": {
        "name": "San Francisco Electric Authority",
        "short_desc": "Official provider for new electricity connections",
        "long_desc": "Government utility responsible for delivering new residential and commercial electricity connections in San Francisco.",
        "additional_desc": {
          "url": "sf_electric_authority"
        },
        "images": [
          {
            "url": "https://sfea.gov/logo.png",
            "size_type": "sm"
          }
        ]
      },
      "short_desc": "Official provider for new electricity connections",
      "fulfillments": [
        {
          "id": "617",
          "type": "SITE_VISIT",
          "rateable": true
        },
        {
          "id": "617",
          "type": "SITE_VISIT",
          "rateable": true
        }
      ],
      "locations": [
        {
          "id": "29",
          "gps": "37.7504,-122.4138",
          "address": "1320 Potrero Ave, SF",
          "city": {
            "name": "San Francisco"
          },
          "country": {
            "name": "United States"
          },
          "state": {
            "name": "CA"
          },
          "area_code": "94110"
        }
      ],
      "rateable": true,
      "items": [
        {
          "id": "471",
          "descriptor": {
            "name": "Residential Electricity Connection",
            "code": "Residential Electricity Connection",
            "short_desc": "Residential Electricity Connection",
            "long_desc": "This residential electricity connection is ideal for homes, apartments, and other living spaces. It ensures a steady power supply suitable for running everyday appliances like lights, fans, TVs, refrigerators, washing machines, and other domestic devices. The connection is limited to moderate load levels and complies with local safety and energy efficiency standards. Perfect for first-time homeowners or tenants seeking reliable electricity.",
            "images": [
              {
                "url": "https://cdn0.iconfinder.com/data/icons/cables-plugs/154/socket-electricity-cable-connect-plug-512.png"
              }
            ]
          },
          "rateable": true,
          "rating": "null",
          "price": {
            "value": "80",
            "currency": "USD",
            "discounted_value": 80.0,
            "reward_based_discount_percent": 0
          },
          "quantity": {
            "available": {
              "count": 20
            }
          },
          "category_ids": [
            "44"
          ],
          "fulfillment_ids": [
            "617"
          ],
          "location_ids": [
            "29"
          ],
          "tags": [
            {
              "descriptor": {
                "description": "connection-type",
                "code": "connection-type"
              },
              "list": [
                {
                  "value": "Resendial",
                  "descriptor": {
                    "code": "property-type",
                    "name": "property-type"
                  },
                  "display": true
                }
              ]
            },
            {
              "descriptor": {
                "description": "documentation",
                "code": "documentation"
              },
              "list": [
                {
                  "value": "Three Phase",
                  "descriptor": {
                    "code": "phase",
                    "name": "phase"
                  },
                  "display": true
                },
                {
                  "value": "Three Phase",
                  "descriptor": {
                    "code": "phase",
                    "name": "phase"
                  },
                  "display": true
                }
              ]
            }
          ]
        },
        {
          "id": "472",
          "descriptor": {
            "name": "Commercial Electricity Connection",
            "code": "Commercial Electricity Connection",
            "short_desc": "Commercial Electricity Connection",
            "long_desc": "Commercial Electricity Connection",
            "images": [
              {
                "url": "https://cdn0.iconfinder.com/data/icons/cables-plugs/154/socket-electricity-cable-connect-plug-512.png"
              }
            ]
          },
          "rateable": true,
          "rating": "null",
          "price": {
            "value": "200",
            "currency": "USD",
            "discounted_value": 200.0,
            "reward_based_discount_percent": 0
          },
          "quantity": {
            "available": {
              "count": 20
            }
          },
          "category_ids": [
            "44"
          ],
          "fulfillment_ids": [
            "617"
          ],
          "location_ids": [
            "29"
          ],
          "tags": [
            {
              "descriptor": {
                "description": "connection-type",
                "code": "connection-type"
              },
              "list": [
                {
                  "value": "Commercial",
                  "descriptor": {
                    "code": "property-type",
                    "name": "property-type"
                  },
                  "display": true
                }
              ]
            },
            {
              "descriptor": {
                "description": "documentation",
                "code": "documentation"
              },
              "list": [
                {
                  "value": "Three Phase",
                  "descriptor": {
                    "code": "phase",
                    "name": "phase"
                  },
                  "display": true
                },
                {
                  "value": "Three Phase",
                  "descriptor": {
                    "code": "phase",
                    "name": "phase"
                  },
                  "display": true
                }
              ]
            }
          ]
        }
      ]
    }
  ]
}

🎁 CATALOG for `Stephanie Stephens` — Reward: -2
{
  "descriptor": {
    "name": "UrjaKart",
    "code": "UrjaKart",
    "short_desc": "A seller platform for energy solutions, enabling seamless transactions of batteries, solar panels, and other renewable energy products."
  },
  "providers": [
    {
      "id": "334",
      "descriptor": {
        "name": "San Francisco Electric Authority",
        "short_desc": "Official provider for new electricity connections",
        "long_desc": "Government utility responsible for delivering new residential and commercial electricity connections in San Francisco.",
        "additional_desc": {
          "url": "sf_electric_authority"
        },
        "images": [
          {
            "url": "https://sfea.gov/logo.png",
            "size_type": "sm"
          }
        ]
      },
      "short_desc": "Official provider for new electricity connections",
      "fulfillments": [
        {
          "id": "617",
          "type": "SITE_VISIT",
          "rateable": true
        },
        {
          "id": "617",
          "type": "SITE_VISIT",
          "rateable": true
        }
      ],
      "locations": [
        {
          "id": "29",
          "gps": "37.7504,-122.4138",
          "address": "1320 Potrero Ave, SF",
          "city": {
            "name": "San Francisco"
          },
          "country": {
            "name": "United States"
          },
          "state": {
            "name": "CA"
          },
          "area_code": "94110"
        }
      ],
      "rateable": true,
      "items": [
        {
          "id": "471",
          "descriptor": {
            "name": "Residential Electricity Connection",
            "code": "Residential Electricity Connection",
            "short_desc": "Residential Electricity Connection",
            "long_desc": "This residential electricity connection is ideal for homes, apartments, and other living spaces. It ensures a steady power supply suitable for running everyday appliances like lights, fans, TVs, refrigerators, washing machines, and other domestic devices. The connection is limited to moderate load levels and complies with local safety and energy efficiency standards. Perfect for first-time homeowners or tenants seeking reliable electricity.",
            "images": [
              {
                "url": "https://cdn0.iconfinder.com/data/icons/cables-plugs/154/socket-electricity-cable-connect-plug-512.png"
              }
            ]
          },
          "rateable": true,
          "rating": "null",
          "price": {
            "value": "80",
            "currency": "USD",
            "discounted_value": 80.0,
            "reward_based_discount_percent": 0
          },
          "quantity": {
            "available": {
              "count": 20
            }
          },
          "category_ids": [
            "44"
          ],
          "fulfillment_ids": [
            "617"
          ],
          "location_ids": [
            "29"
          ],
          "tags": [
            {
              "descriptor": {
                "description": "connection-type",
                "code": "connection-type"
              },
              "list": [
                {
                  "value": "Resendial",
                  "descriptor": {
                    "code": "property-type",
                    "name": "property-type"
                  },
                  "display": true
                }
              ]
            },
            {
              "descriptor": {
                "description": "documentation",
                "code": "documentation"
              },
              "list": [
                {
                  "value": "Three Phase",
                  "descriptor": {
                    "code": "phase",
                    "name": "phase"
                  },
                  "display": true
                },
                {
                  "value": "Three Phase",
                  "descriptor": {
                    "code": "phase",
                    "name": "phase"
                  },
                  "display": true
                }
              ]
            }
          ]
        },
        {
          "id": "472",
          "descriptor": {
            "name": "Commercial Electricity Connection",
            "code": "Commercial Electricity Connection",
            "short_desc": "Commercial Electricity Connection",
            "long_desc": "Commercial Electricity Connection",
            "images": [
              {
                "url": "https://cdn0.iconfinder.com/data/icons/cables-plugs/154/socket-electricity-cable-connect-plug-512.png"
              }
            ]
          },
          "rateable": true,
          "rating": "null",
          "price": {
            "value": "200",
            "currency": "USD",
            "discounted_value": 200.0,
            "reward_based_discount_percent": 0
          },
          "quantity": {
            "available": {
              "count": 20
            }
          },
          "category_ids": [
            "44"
          ],
          "fulfillment_ids": [
            "617"
          ],
          "location_ids": [
            "29"
          ],
          "tags": [
            {
              "descriptor": {
                "description": "connection-type",
                "code": "connection-type"
              },
              "list": [
                {
                  "value": "Commercial",
                  "descriptor": {
                    "code": "property-type",
                    "name": "property-type"
                  },
                  "display": true
                }
              ]
            },
            {
              "descriptor": {
                "description": "documentation",
                "code": "documentation"
              },
              "list": [
                {
                  "value": "Three Phase",
                  "descriptor": {
                    "code": "phase",
                    "name": "phase"
                  },
                  "display": true
                },
                {
                  "value": "Three Phase",
                  "descriptor": {
                    "code": "phase",
                    "name": "phase"
                  },
                  "display": true
                }
              ]
            }
          ]
        }
      ]
    }
  ]
}

🎁 CATALOG for `Lisa Kim` — Reward: 5
{
  "descriptor": {
    "name": "UrjaKart",
    "code": "UrjaKart",
    "short_desc": "A seller platform for energy solutions, enabling seamless transactions of batteries, solar panels, and other renewable energy products."
  },
  "providers": [
    {
      "id": "334",
      "descriptor": {
        "name": "San Francisco Electric Authority",
        "short_desc": "Official provider for new electricity connections",
        "long_desc": "Government utility responsible for delivering new residential and commercial electricity connections in San Francisco.",
        "additional_desc": {
          "url": "sf_electric_authority"
        },
        "images": [
          {
            "url": "https://sfea.gov/logo.png",
            "size_type": "sm"
          }
        ]
      },
      "short_desc": "Official provider for new electricity connections",
      "fulfillments": [
        {
          "id": "617",
          "type": "SITE_VISIT",
          "rateable": true
        },
        {
          "id": "617",
          "type": "SITE_VISIT",
          "rateable": true
        }
      ],
      "locations": [
        {
          "id": "29",
          "gps": "37.7504,-122.4138",
          "address": "1320 Potrero Ave, SF",
          "city": {
            "name": "San Francisco"
          },
          "country": {
            "name": "United States"
          },
          "state": {
            "name": "CA"
          },
          "area_code": "94110"
        }
      ],
      "rateable": true,
      "items": [
        {
          "id": "471",
          "descriptor": {
            "name": "Residential Electricity Connection",
            "code": "Residential Electricity Connection",
            "short_desc": "Residential Electricity Connection",
            "long_desc": "This residential electricity connection is ideal for homes, apartments, and other living spaces. It ensures a steady power supply suitable for running everyday appliances like lights, fans, TVs, refrigerators, washing machines, and other domestic devices. The connection is limited to moderate load levels and complies with local safety and energy efficiency standards. Perfect for first-time homeowners or tenants seeking reliable electricity.",
            "images": [
              {
                "url": "https://cdn0.iconfinder.com/data/icons/cables-plugs/154/socket-electricity-cable-connect-plug-512.png"
              }
            ]
          },
          "rateable": true,
          "rating": "null",
          "price": {
            "value": "80",
            "currency": "USD",
            "discounted_value": 64.0,
            "reward_based_discount_percent": 20
          },
          "quantity": {
            "available": {
              "count": 20
            }
          },
          "category_ids": [
            "44"
          ],
          "fulfillment_ids": [
            "617"
          ],
          "location_ids": [
            "29"
          ],
          "tags": [
            {
              "descriptor": {
                "description": "connection-type",
                "code": "connection-type"
              },
              "list": [
                {
                  "value": "Resendial",
                  "descriptor": {
                    "code": "property-type",
                    "name": "property-type"
                  },
                  "display": true
                }
              ]
            },
            {
              "descriptor": {
                "description": "documentation",
                "code": "documentation"
              },
              "list": [
                {
                  "value": "Three Phase",
                  "descriptor": {
                    "code": "phase",
                    "name": "phase"
                  },
                  "display": true
                },
                {
                  "value": "Three Phase",
                  "descriptor": {
                    "code": "phase",
                    "name": "phase"
                  },
                  "display": true
                }
              ]
            }
          ]
        },
        {
          "id": "472",
          "descriptor": {
            "name": "Commercial Electricity Connection",
            "code": "Commercial Electricity Connection",
            "short_desc": "Commercial Electricity Connection",
            "long_desc": "Commercial Electricity Connection",
            "images": [
              {
                "url": "https://cdn0.iconfinder.com/data/icons/cables-plugs/154/socket-electricity-cable-connect-plug-512.png"
              }
            ]
          },
          "rateable": true,
          "rating": "null",
          "price": {
            "value": "200",
            "currency": "USD",
            "discounted_value": 160.0,
            "reward_based_discount_percent": 20
          },
          "quantity": {
            "available": {
              "count": 20
            }
          },
          "category_ids": [
            "44"
          ],
          "fulfillment_ids": [
            "617"
          ],
          "location_ids": [
            "29"
          ],
          "tags": [
            {
              "descriptor": {
                "description": "connection-type",
                "code": "connection-type"
              },
              "list": [
                {
                  "value": "Commercial",
                  "descriptor": {
                    "code": "property-type",
                    "name": "property-type"
                  },
                  "display": true
                }
              ]
            },
            {
              "descriptor": {
                "description": "documentation",
                "code": "documentation"
              },
              "list": [
                {
                  "value": "Three Phase",
                  "descriptor": {
                    "code": "phase",
                    "name": "phase"
                  },
                  "display": true
                },
                {
                  "value": "Three Phase",
                  "descriptor": {
                    "code": "phase",
                    "name": "phase"
                  },
                  "display": true
                }
              ]
            }
          ]
        }
      ]
    }
  ]
}

🎁 CATALOG for `Thomas Hughes` — Reward: 2
{
  "descriptor": {
    "name": "UrjaKart",
    "code": "UrjaKart",
    "short_desc": "A seller platform for energy solutions, enabling seamless transactions of batteries, solar panels, and other renewable energy products."
  },
  "providers": [
    {
      "id": "334",
      "descriptor": {
        "name": "San Francisco Electric Authority",
        "short_desc": "Official provider for new electricity connections",
        "long_desc": "Government utility responsible for delivering new residential and commercial electricity connections in San Francisco.",
        "additional_desc": {
          "url": "sf_electric_authority"
        },
        "images": [
          {
            "url": "https://sfea.gov/logo.png",
            "size_type": "sm"
          }
        ]
      },
      "short_desc": "Official provider for new electricity connections",
      "fulfillments": [
        {
          "id": "617",
          "type": "SITE_VISIT",
          "rateable": true
        },
        {
          "id": "617",
          "type": "SITE_VISIT",
          "rateable": true
        }
      ],
      "locations": [
        {
          "id": "29",
          "gps": "37.7504,-122.4138",
          "address": "1320 Potrero Ave, SF",
          "city": {
            "name": "San Francisco"
          },
          "country": {
            "name": "United States"
          },
          "state": {
            "name": "CA"
          },
          "area_code": "94110"
        }
      ],
      "rateable": true,
      "items": [
        {
          "id": "471",
          "descriptor": {
            "name": "Residential Electricity Connection",
            "code": "Residential Electricity Connection",
            "short_desc": "Residential Electricity Connection",
            "long_desc": "This residential electricity connection is ideal for homes, apartments, and other living spaces. It ensures a steady power supply suitable for running everyday appliances like lights, fans, TVs, refrigerators, washing machines, and other domestic devices. The connection is limited to moderate load levels and complies with local safety and energy efficiency standards. Perfect for first-time homeowners or tenants seeking reliable electricity.",
            "images": [
              {
                "url": "https://cdn0.iconfinder.com/data/icons/cables-plugs/154/socket-electricity-cable-connect-plug-512.png"
              }
            ]
          },
          "rateable": true,
          "rating": "null",
          "price": {
            "value": "80",
            "currency": "USD",
            "discounted_value": 72.0,
            "reward_based_discount_percent": 10
          },
          "quantity": {
            "available": {
              "count": 20
            }
          },
          "category_ids": [
            "44"
          ],
          "fulfillment_ids": [
            "617"
          ],
          "location_ids": [
            "29"
          ],
          "tags": [
            {
              "descriptor": {
                "description": "connection-type",
                "code": "connection-type"
              },
              "list": [
                {
                  "value": "Resendial",
                  "descriptor": {
                    "code": "property-type",
                    "name": "property-type"
                  },
                  "display": true
                }
              ]
            },
            {
              "descriptor": {
                "description": "documentation",
                "code": "documentation"
              },
              "list": [
                {
                  "value": "Three Phase",
                  "descriptor": {
                    "code": "phase",
                    "name": "phase"
                  },
                  "display": true
                },
                {
                  "value": "Three Phase",
                  "descriptor": {
                    "code": "phase",
                    "name": "phase"
                  },
                  "display": true
                }
              ]
            }
          ]
        },
        {
          "id": "472",
          "descriptor": {
            "name": "Commercial Electricity Connection",
            "code": "Commercial Electricity Connection",
            "short_desc": "Commercial Electricity Connection",
            "long_desc": "Commercial Electricity Connection",
            "images": [
              {
                "url": "https://cdn0.iconfinder.com/data/icons/cables-plugs/154/socket-electricity-cable-connect-plug-512.png"
              }
            ]
          },
          "rateable": true,
          "rating": "null",
          "price": {
            "value": "200",
            "currency": "USD",
            "discounted_value": 180.0,
            "reward_based_discount_percent": 10
          },
          "quantity": {
            "available": {
              "count": 20
            }
          },
          "category_ids": [
            "44"
          ],
          "fulfillment_ids": [
            "617"
          ],
          "location_ids": [
            "29"
          ],
          "tags": [
            {
              "descriptor": {
                "description": "connection-type",
                "code": "connection-type"
              },
              "list": [
                {
                  "value": "Commercial",
                  "descriptor": {
                    "code": "property-type",
                    "name": "property-type"
                  },
                  "display": true
                }
              ]
            },
            {
              "descriptor": {
                "description": "documentation",
                "code": "documentation"
              },
              "list": [
                {
                  "value": "Three Phase",
                  "descriptor": {
                    "code": "phase",
                    "name": "phase"
                  },
                  "display": true
                },
                {
                  "value": "Three Phase",
                  "descriptor": {
                    "code": "phase",
                    "name": "phase"
                  },
                  "display": true
                }
              ]
            }
          ]
        }
      ]
    }
  ]
}

🎁 CATALOG for `Adam Conley` — Reward: -1
{
  "descriptor": {
    "name": "UrjaKart",
    "code": "UrjaKart",
    "short_desc": "A seller platform for energy solutions, enabling seamless transactions of batteries, solar panels, and other renewable energy products."
  },
  "providers": [
    {
      "id": "334",
      "descriptor": {
        "name": "San Francisco Electric Authority",
        "short_desc": "Official provider for new electricity connections",
        "long_desc": "Government utility responsible for delivering new residential and commercial electricity connections in San Francisco.",
        "additional_desc": {
          "url": "sf_electric_authority"
        },
        "images": [
          {
            "url": "https://sfea.gov/logo.png",
            "size_type": "sm"
          }
        ]
      },
      "short_desc": "Official provider for new electricity connections",
      "fulfillments": [
        {
          "id": "617",
          "type": "SITE_VISIT",
          "rateable": true
        },
        {
          "id": "617",
          "type": "SITE_VISIT",
          "rateable": true
        }
      ],
      "locations": [
        {
          "id": "29",
          "gps": "37.7504,-122.4138",
          "address": "1320 Potrero Ave, SF",
          "city": {
            "name": "San Francisco"
          },
          "country": {
            "name": "United States"
          },
          "state": {
            "name": "CA"
          },
          "area_code": "94110"
        }
      ],
      "rateable": true,
      "items": [
        {
          "id": "471",
          "descriptor": {
            "name": "Residential Electricity Connection",
            "code": "Residential Electricity Connection",
            "short_desc": "Residential Electricity Connection",
            "long_desc": "This residential electricity connection is ideal for homes, apartments, and other living spaces. It ensures a steady power supply suitable for running everyday appliances like lights, fans, TVs, refrigerators, washing machines, and other domestic devices. The connection is limited to moderate load levels and complies with local safety and energy efficiency standards. Perfect for first-time homeowners or tenants seeking reliable electricity.",
            "images": [
              {
                "url": "https://cdn0.iconfinder.com/data/icons/cables-plugs/154/socket-electricity-cable-connect-plug-512.png"
              }
            ]
          },
          "rateable": true,
          "rating": "null",
          "price": {
            "value": "80",
            "currency": "USD",
            "discounted_value": 80.0,
            "reward_based_discount_percent": 0
          },
          "quantity": {
            "available": {
              "count": 20
            }
          },
          "category_ids": [
            "44"
          ],
          "fulfillment_ids": [
            "617"
          ],
          "location_ids": [
            "29"
          ],
          "tags": [
            {
              "descriptor": {
                "description": "connection-type",
                "code": "connection-type"
              },
              "list": [
                {
                  "value": "Resendial",
                  "descriptor": {
                    "code": "property-type",
                    "name": "property-type"
                  },
                  "display": true
                }
              ]
            },
            {
              "descriptor": {
                "description": "documentation",
                "code": "documentation"
              },
              "list": [
                {
                  "value": "Three Phase",
                  "descriptor": {
                    "code": "phase",
                    "name": "phase"
                  },
                  "display": true
                },
                {
                  "value": "Three Phase",
                  "descriptor": {
                    "code": "phase",
                    "name": "phase"
                  },
                  "display": true
                }
              ]
            }
          ]
        },
        {
          "id": "472",
          "descriptor": {
            "name": "Commercial Electricity Connection",
            "code": "Commercial Electricity Connection",
            "short_desc": "Commercial Electricity Connection",
            "long_desc": "Commercial Electricity Connection",
            "images": [
              {
                "url": "https://cdn0.iconfinder.com/data/icons/cables-plugs/154/socket-electricity-cable-connect-plug-512.png"
              }
            ]
          },
          "rateable": true,
          "rating": "null",
          "price": {
            "value": "200",
            "currency": "USD",
            "discounted_value": 200.0,
            "reward_based_discount_percent": 0
          },
          "quantity": {
            "available": {
              "count": 20
            }
          },
          "category_ids": [
            "44"
          ],
          "fulfillment_ids": [
            "617"
          ],
          "location_ids": [
            "29"
          ],
          "tags": [
            {
              "descriptor": {
                "description": "connection-type",
                "code": "connection-type"
              },
              "list": [
                {
                  "value": "Commercial",
                  "descriptor": {
                    "code": "property-type",
                    "name": "property-type"
                  },
                  "display": true
                }
              ]
            },
            {
              "descriptor": {
                "description": "documentation",
                "code": "documentation"
              },
              "list": [
                {
                  "value": "Three Phase",
                  "descriptor": {
                    "code": "phase",
                    "name": "phase"
                  },
                  "display": true
                },
                {
                  "value": "Three Phase",
                  "descriptor": {
                    "code": "phase",
                    "name": "phase"
                  },
                  "display": true
                }
              ]
            }
          ]
        }
      ]
    }
  ]
}

🎁 CATALOG for `Hunter Williams` — Reward: -1
{
  "descriptor": {
    "name": "UrjaKart",
    "code": "UrjaKart",
    "short_desc": "A seller platform for energy solutions, enabling seamless transactions of batteries, solar panels, and other renewable energy products."
  },
  "providers": [
    {
      "id": "334",
      "descriptor": {
        "name": "San Francisco Electric Authority",
        "short_desc": "Official provider for new electricity connections",
        "long_desc": "Government utility responsible for delivering new residential and commercial electricity connections in San Francisco.",
        "additional_desc": {
          "url": "sf_electric_authority"
        },
        "images": [
          {
            "url": "https://sfea.gov/logo.png",
            "size_type": "sm"
          }
        ]
      },
      "short_desc": "Official provider for new electricity connections",
      "fulfillments": [
        {
          "id": "617",
          "type": "SITE_VISIT",
          "rateable": true
        },
        {
          "id": "617",
          "type": "SITE_VISIT",
          "rateable": true
        }
      ],
      "locations": [
        {
          "id": "29",
          "gps": "37.7504,-122.4138",
          "address": "1320 Potrero Ave, SF",
          "city": {
            "name": "San Francisco"
          },
          "country": {
            "name": "United States"
          },
          "state": {
            "name": "CA"
          },
          "area_code": "94110"
        }
      ],
      "rateable": true,
      "items": [
        {
          "id": "471",
          "descriptor": {
            "name": "Residential Electricity Connection",
            "code": "Residential Electricity Connection",
            "short_desc": "Residential Electricity Connection",
            "long_desc": "This residential electricity connection is ideal for homes, apartments, and other living spaces. It ensures a steady power supply suitable for running everyday appliances like lights, fans, TVs, refrigerators, washing machines, and other domestic devices. The connection is limited to moderate load levels and complies with local safety and energy efficiency standards. Perfect for first-time homeowners or tenants seeking reliable electricity.",
            "images": [
              {
                "url": "https://cdn0.iconfinder.com/data/icons/cables-plugs/154/socket-electricity-cable-connect-plug-512.png"
              }
            ]
          },
          "rateable": true,
          "rating": "null",
          "price": {
            "value": "80",
            "currency": "USD",
            "discounted_value": 80.0,
            "reward_based_discount_percent": 0
          },
          "quantity": {
            "available": {
              "count": 20
            }
          },
          "category_ids": [
            "44"
          ],
          "fulfillment_ids": [
            "617"
          ],
          "location_ids": [
            "29"
          ],
          "tags": [
            {
              "descriptor": {
                "description": "connection-type",
                "code": "connection-type"
              },
              "list": [
                {
                  "value": "Resendial",
                  "descriptor": {
                    "code": "property-type",
                    "name": "property-type"
                  },
                  "display": true
                }
              ]
            },
            {
              "descriptor": {
                "description": "documentation",
                "code": "documentation"
              },
              "list": [
                {
                  "value": "Three Phase",
                  "descriptor": {
                    "code": "phase",
                    "name": "phase"
                  },
                  "display": true
                },
                {
                  "value": "Three Phase",
                  "descriptor": {
                    "code": "phase",
                    "name": "phase"
                  },
                  "display": true
                }
              ]
            }
          ]
        },
        {
          "id": "472",
          "descriptor": {
            "name": "Commercial Electricity Connection",
            "code": "Commercial Electricity Connection",
            "short_desc": "Commercial Electricity Connection",
            "long_desc": "Commercial Electricity Connection",
            "images": [
              {
                "url": "https://cdn0.iconfinder.com/data/icons/cables-plugs/154/socket-electricity-cable-connect-plug-512.png"
              }
            ]
          },
          "rateable": true,
          "rating": "null",
          "price": {
            "value": "200",
            "currency": "USD",
            "discounted_value": 200.0,
            "reward_based_discount_percent": 0
          },
          "quantity": {
            "available": {
              "count": 20
            }
          },
          "category_ids": [
            "44"
          ],
          "fulfillment_ids": [
            "617"
          ],
          "location_ids": [
            "29"
          ],
          "tags": [
            {
              "descriptor": {
                "description": "connection-type",
                "code": "connection-type"
              },
              "list": [
                {
                  "value": "Commercial",
                  "descriptor": {
                    "code": "property-type",
                    "name": "property-type"
                  },
                  "display": true
                }
              ]
            },
            {
              "descriptor": {
                "description": "documentation",
                "code": "documentation"
              },
              "list": [
                {
                  "value": "Three Phase",
                  "descriptor": {
                    "code": "phase",
                    "name": "phase"
                  },
                  "display": true
                },
                {
                  "value": "Three Phase",
                  "descriptor": {
                    "code": "phase",
                    "name": "phase"
                  },
                  "display": true
                }
              ]
            }
          ]
        }
      ]
    }
  ]
}

🎁 CATALOG for `Pamela Garcia` — Reward: 2
{
  "descriptor": {
    "name": "UrjaKart",
    "code": "UrjaKart",
    "short_desc": "A seller platform for energy solutions, enabling seamless transactions of batteries, solar panels, and other renewable energy products."
  },
  "providers": [
    {
      "id": "334",
      "descriptor": {
        "name": "San Francisco Electric Authority",
        "short_desc": "Official provider for new electricity connections",
        "long_desc": "Government utility responsible for delivering new residential and commercial electricity connections in San Francisco.",
        "additional_desc": {
          "url": "sf_electric_authority"
        },
        "images": [
          {
            "url": "https://sfea.gov/logo.png",
            "size_type": "sm"
          }
        ]
      },
      "short_desc": "Official provider for new electricity connections",
      "fulfillments": [
        {
          "id": "617",
          "type": "SITE_VISIT",
          "rateable": true
        },
        {
          "id": "617",
          "type": "SITE_VISIT",
          "rateable": true
        }
      ],
      "locations": [
        {
          "id": "29",
          "gps": "37.7504,-122.4138",
          "address": "1320 Potrero Ave, SF",
          "city": {
            "name": "San Francisco"
          },
          "country": {
            "name": "United States"
          },
          "state": {
            "name": "CA"
          },
          "area_code": "94110"
        }
      ],
      "rateable": true,
      "items": [
        {
          "id": "471",
          "descriptor": {
            "name": "Residential Electricity Connection",
            "code": "Residential Electricity Connection",
            "short_desc": "Residential Electricity Connection",
            "long_desc": "This residential electricity connection is ideal for homes, apartments, and other living spaces. It ensures a steady power supply suitable for running everyday appliances like lights, fans, TVs, refrigerators, washing machines, and other domestic devices. The connection is limited to moderate load levels and complies with local safety and energy efficiency standards. Perfect for first-time homeowners or tenants seeking reliable electricity.",
            "images": [
              {
                "url": "https://cdn0.iconfinder.com/data/icons/cables-plugs/154/socket-electricity-cable-connect-plug-512.png"
              }
            ]
          },
          "rateable": true,
          "rating": "null",
          "price": {
            "value": "80",
            "currency": "USD",
            "discounted_value": 72.0,
            "reward_based_discount_percent": 10
          },
          "quantity": {
            "available": {
              "count": 20
            }
          },
          "category_ids": [
            "44"
          ],
          "fulfillment_ids": [
            "617"
          ],
          "location_ids": [
            "29"
          ],
          "tags": [
            {
              "descriptor": {
                "description": "connection-type",
                "code": "connection-type"
              },
              "list": [
                {
                  "value": "Resendial",
                  "descriptor": {
                    "code": "property-type",
                    "name": "property-type"
                  },
                  "display": true
                }
              ]
            },
            {
              "descriptor": {
                "description": "documentation",
                "code": "documentation"
              },
              "list": [
                {
                  "value": "Three Phase",
                  "descriptor": {
                    "code": "phase",
                    "name": "phase"
                  },
                  "display": true
                },
                {
                  "value": "Three Phase",
                  "descriptor": {
                    "code": "phase",
                    "name": "phase"
                  },
                  "display": true
                }
              ]
            }
          ]
        },
        {
          "id": "472",
          "descriptor": {
            "name": "Commercial Electricity Connection",
            "code": "Commercial Electricity Connection",
            "short_desc": "Commercial Electricity Connection",
            "long_desc": "Commercial Electricity Connection",
            "images": [
              {
                "url": "https://cdn0.iconfinder.com/data/icons/cables-plugs/154/socket-electricity-cable-connect-plug-512.png"
              }
            ]
          },
          "rateable": true,
          "rating": "null",
          "price": {
            "value": "200",
            "currency": "USD",
            "discounted_value": 180.0,
            "reward_based_discount_percent": 10
          },
          "quantity": {
            "available": {
              "count": 20
            }
          },
          "category_ids": [
            "44"
          ],
          "fulfillment_ids": [
            "617"
          ],
          "location_ids": [
            "29"
          ],
          "tags": [
            {
              "descriptor": {
                "description": "connection-type",
                "code": "connection-type"
              },
              "list": [
                {
                  "value": "Commercial",
                  "descriptor": {
                    "code": "property-type",
                    "name": "property-type"
                  },
                  "display": true
                }
              ]
            },
            {
              "descriptor": {
                "description": "documentation",
                "code": "documentation"
              },
              "list": [
                {
                  "value": "Three Phase",
                  "descriptor": {
                    "code": "phase",
                    "name": "phase"
                  },
                  "display": true
                },
                {
                  "value": "Three Phase",
                  "descriptor": {
                    "code": "phase",
                    "name": "phase"
                  },
                  "display": true
                }
              ]
            }
          ]
        }
      ]
    }
  ]
}

🎁 CATALOG for `Fernando Little` — Reward: -1
{
  "descriptor": {
    "name": "UrjaKart",
    "code": "UrjaKart",
    "short_desc": "A seller platform for energy solutions, enabling seamless transactions of batteries, solar panels, and other renewable energy products."
  },
  "providers": [
    {
      "id": "334",
      "descriptor": {
        "name": "San Francisco Electric Authority",
        "short_desc": "Official provider for new electricity connections",
        "long_desc": "Government utility responsible for delivering new residential and commercial electricity connections in San Francisco.",
        "additional_desc": {
          "url": "sf_electric_authority"
        },
        "images": [
          {
            "url": "https://sfea.gov/logo.png",
            "size_type": "sm"
          }
        ]
      },
      "short_desc": "Official provider for new electricity connections",
      "fulfillments": [
        {
          "id": "617",
          "type": "SITE_VISIT",
          "rateable": true
        },
        {
          "id": "617",
          "type": "SITE_VISIT",
          "rateable": true
        }
      ],
      "locations": [
        {
          "id": "29",
          "gps": "37.7504,-122.4138",
          "address": "1320 Potrero Ave, SF",
          "city": {
            "name": "San Francisco"
          },
          "country": {
            "name": "United States"
          },
          "state": {
            "name": "CA"
          },
          "area_code": "94110"
        }
      ],
      "rateable": true,
      "items": [
        {
          "id": "471",
          "descriptor": {
            "name": "Residential Electricity Connection",
            "code": "Residential Electricity Connection",
            "short_desc": "Residential Electricity Connection",
            "long_desc": "This residential electricity connection is ideal for homes, apartments, and other living spaces. It ensures a steady power supply suitable for running everyday appliances like lights, fans, TVs, refrigerators, washing machines, and other domestic devices. The connection is limited to moderate load levels and complies with local safety and energy efficiency standards. Perfect for first-time homeowners or tenants seeking reliable electricity.",
            "images": [
              {
                "url": "https://cdn0.iconfinder.com/data/icons/cables-plugs/154/socket-electricity-cable-connect-plug-512.png"
              }
            ]
          },
          "rateable": true,
          "rating": "null",
          "price": {
            "value": "80",
            "currency": "USD",
            "discounted_value": 80.0,
            "reward_based_discount_percent": 0
          },
          "quantity": {
            "available": {
              "count": 20
            }
          },
          "category_ids": [
            "44"
          ],
          "fulfillment_ids": [
            "617"
          ],
          "location_ids": [
            "29"
          ],
          "tags": [
            {
              "descriptor": {
                "description": "connection-type",
                "code": "connection-type"
              },
              "list": [
                {
                  "value": "Resendial",
                  "descriptor": {
                    "code": "property-type",
                    "name": "property-type"
                  },
                  "display": true
                }
              ]
            },
            {
              "descriptor": {
                "description": "documentation",
                "code": "documentation"
              },
              "list": [
                {
                  "value": "Three Phase",
                  "descriptor": {
                    "code": "phase",
                    "name": "phase"
                  },
                  "display": true
                },
                {
                  "value": "Three Phase",
                  "descriptor": {
                    "code": "phase",
                    "name": "phase"
                  },
                  "display": true
                }
              ]
            }
          ]
        },
        {
          "id": "472",
          "descriptor": {
            "name": "Commercial Electricity Connection",
            "code": "Commercial Electricity Connection",
            "short_desc": "Commercial Electricity Connection",
            "long_desc": "Commercial Electricity Connection",
            "images": [
              {
                "url": "https://cdn0.iconfinder.com/data/icons/cables-plugs/154/socket-electricity-cable-connect-plug-512.png"
              }
            ]
          },
          "rateable": true,
          "rating": "null",
          "price": {
            "value": "200",
            "currency": "USD",
            "discounted_value": 200.0,
            "reward_based_discount_percent": 0
          },
          "quantity": {
            "available": {
              "count": 20
            }
          },
          "category_ids": [
            "44"
          ],
          "fulfillment_ids": [
            "617"
          ],
          "location_ids": [
            "29"
          ],
          "tags": [
            {
              "descriptor": {
                "description": "connection-type",
                "code": "connection-type"
              },
              "list": [
                {
                  "value": "Commercial",
                  "descriptor": {
                    "code": "property-type",
                    "name": "property-type"
                  },
                  "display": true
                }
              ]
            },
            {
              "descriptor": {
                "description": "documentation",
                "code": "documentation"
              },
              "list": [
                {
                  "value": "Three Phase",
                  "descriptor": {
                    "code": "phase",
                    "name": "phase"
                  },
                  "display": true
                },
                {
                  "value": "Three Phase",
                  "descriptor": {
                    "code": "phase",
                    "name": "phase"
                  },
                  "display": true
                }
              ]
            }
          ]
        }
      ]
    }
  ]
}

🎁 CATALOG for `Misty Rubio MD` — Reward: -2
{
  "descriptor": {
    "name": "UrjaKart",
    "code": "UrjaKart",
    "short_desc": "A seller platform for energy solutions, enabling seamless transactions of batteries, solar panels, and other renewable energy products."
  },
  "providers": [
    {
      "id": "334",
      "descriptor": {
        "name": "San Francisco Electric Authority",
        "short_desc": "Official provider for new electricity connections",
        "long_desc": "Government utility responsible for delivering new residential and commercial electricity connections in San Francisco.",
        "additional_desc": {
          "url": "sf_electric_authority"
        },
        "images": [
          {
            "url": "https://sfea.gov/logo.png",
            "size_type": "sm"
          }
        ]
      },
      "short_desc": "Official provider for new electricity connections",
      "fulfillments": [
        {
          "id": "617",
          "type": "SITE_VISIT",
          "rateable": true
        },
        {
          "id": "617",
          "type": "SITE_VISIT",
          "rateable": true
        }
      ],
      "locations": [
        {
          "id": "29",
          "gps": "37.7504,-122.4138",
          "address": "1320 Potrero Ave, SF",
          "city": {
            "name": "San Francisco"
          },
          "country": {
            "name": "United States"
          },
          "state": {
            "name": "CA"
          },
          "area_code": "94110"
        }
      ],
      "rateable": true,
      "items": [
        {
          "id": "471",
          "descriptor": {
            "name": "Residential Electricity Connection",
            "code": "Residential Electricity Connection",
            "short_desc": "Residential Electricity Connection",
            "long_desc": "This residential electricity connection is ideal for homes, apartments, and other living spaces. It ensures a steady power supply suitable for running everyday appliances like lights, fans, TVs, refrigerators, washing machines, and other domestic devices. The connection is limited to moderate load levels and complies with local safety and energy efficiency standards. Perfect for first-time homeowners or tenants seeking reliable electricity.",
            "images": [
              {
                "url": "https://cdn0.iconfinder.com/data/icons/cables-plugs/154/socket-electricity-cable-connect-plug-512.png"
              }
            ]
          },
          "rateable": true,
          "rating": "null",
          "price": {
            "value": "80",
            "currency": "USD",
            "discounted_value": 80.0,
            "reward_based_discount_percent": 0
          },
          "quantity": {
            "available": {
              "count": 20
            }
          },
          "category_ids": [
            "44"
          ],
          "fulfillment_ids": [
            "617"
          ],
          "location_ids": [
            "29"
          ],
          "tags": [
            {
              "descriptor": {
                "description": "connection-type",
                "code": "connection-type"
              },
              "list": [
                {
                  "value": "Resendial",
                  "descriptor": {
                    "code": "property-type",
                    "name": "property-type"
                  },
                  "display": true
                }
              ]
            },
            {
              "descriptor": {
                "description": "documentation",
                "code": "documentation"
              },
              "list": [
                {
                  "value": "Three Phase",
                  "descriptor": {
                    "code": "phase",
                    "name": "phase"
                  },
                  "display": true
                },
                {
                  "value": "Three Phase",
                  "descriptor": {
                    "code": "phase",
                    "name": "phase"
                  },
                  "display": true
                }
              ]
            }
          ]
        },
        {
          "id": "472",
          "descriptor": {
            "name": "Commercial Electricity Connection",
            "code": "Commercial Electricity Connection",
            "short_desc": "Commercial Electricity Connection",
            "long_desc": "Commercial Electricity Connection",
            "images": [
              {
                "url": "https://cdn0.iconfinder.com/data/icons/cables-plugs/154/socket-electricity-cable-connect-plug-512.png"
              }
            ]
          },
          "rateable": true,
          "rating": "null",
          "price": {
            "value": "200",
            "currency": "USD",
            "discounted_value": 200.0,
            "reward_based_discount_percent": 0
          },
          "quantity": {
            "available": {
              "count": 20
            }
          },
          "category_ids": [
            "44"
          ],
          "fulfillment_ids": [
            "617"
          ],
          "location_ids": [
            "29"
          ],
          "tags": [
            {
              "descriptor": {
                "description": "connection-type",
                "code": "connection-type"
              },
              "list": [
                {
                  "value": "Commercial",
                  "descriptor": {
                    "code": "property-type",
                    "name": "property-type"
                  },
                  "display": true
                }
              ]
            },
            {
              "descriptor": {
                "description": "documentation",
                "code": "documentation"
              },
              "list": [
                {
                  "value": "Three Phase",
                  "descriptor": {
                    "code": "phase",
                    "name": "phase"
                  },
                  "display": true
                },
                {
                  "value": "Three Phase",
                  "descriptor": {
                    "code": "phase",
                    "name": "phase"
                  },
                  "display": true
                }
              ]
            }
          ]
        }
      ]
    }
  ]
}

🎁 CATALOG for `Mary Vazquez` — Reward: 2
{
  "descriptor": {
    "name": "UrjaKart",
    "code": "UrjaKart",
    "short_desc": "A seller platform for energy solutions, enabling seamless transactions of batteries, solar panels, and other renewable energy products."
  },
  "providers": [
    {
      "id": "334",
      "descriptor": {
        "name": "San Francisco Electric Authority",
        "short_desc": "Official provider for new electricity connections",
        "long_desc": "Government utility responsible for delivering new residential and commercial electricity connections in San Francisco.",
        "additional_desc": {
          "url": "sf_electric_authority"
        },
        "images": [
          {
            "url": "https://sfea.gov/logo.png",
            "size_type": "sm"
          }
        ]
      },
      "short_desc": "Official provider for new electricity connections",
      "fulfillments": [
        {
          "id": "617",
          "type": "SITE_VISIT",
          "rateable": true
        },
        {
          "id": "617",
          "type": "SITE_VISIT",
          "rateable": true
        }
      ],
      "locations": [
        {
          "id": "29",
          "gps": "37.7504,-122.4138",
          "address": "1320 Potrero Ave, SF",
          "city": {
            "name": "San Francisco"
          },
          "country": {
            "name": "United States"
          },
          "state": {
            "name": "CA"
          },
          "area_code": "94110"
        }
      ],
      "rateable": true,
      "items": [
        {
          "id": "471",
          "descriptor": {
            "name": "Residential Electricity Connection",
            "code": "Residential Electricity Connection",
            "short_desc": "Residential Electricity Connection",
            "long_desc": "This residential electricity connection is ideal for homes, apartments, and other living spaces. It ensures a steady power supply suitable for running everyday appliances like lights, fans, TVs, refrigerators, washing machines, and other domestic devices. The connection is limited to moderate load levels and complies with local safety and energy efficiency standards. Perfect for first-time homeowners or tenants seeking reliable electricity.",
            "images": [
              {
                "url": "https://cdn0.iconfinder.com/data/icons/cables-plugs/154/socket-electricity-cable-connect-plug-512.png"
              }
            ]
          },
          "rateable": true,
          "rating": "null",
          "price": {
            "value": "80",
            "currency": "USD",
            "discounted_value": 72.0,
            "reward_based_discount_percent": 10
          },
          "quantity": {
            "available": {
              "count": 20
            }
          },
          "category_ids": [
            "44"
          ],
          "fulfillment_ids": [
            "617"
          ],
          "location_ids": [
            "29"
          ],
          "tags": [
            {
              "descriptor": {
                "description": "connection-type",
                "code": "connection-type"
              },
              "list": [
                {
                  "value": "Resendial",
                  "descriptor": {
                    "code": "property-type",
                    "name": "property-type"
                  },
                  "display": true
                }
              ]
            },
            {
              "descriptor": {
                "description": "documentation",
                "code": "documentation"
              },
              "list": [
                {
                  "value": "Three Phase",
                  "descriptor": {
                    "code": "phase",
                    "name": "phase"
                  },
                  "display": true
                },
                {
                  "value": "Three Phase",
                  "descriptor": {
                    "code": "phase",
                    "name": "phase"
                  },
                  "display": true
                }
              ]
            }
          ]
        },
        {
          "id": "472",
          "descriptor": {
            "name": "Commercial Electricity Connection",
            "code": "Commercial Electricity Connection",
            "short_desc": "Commercial Electricity Connection",
            "long_desc": "Commercial Electricity Connection",
            "images": [
              {
                "url": "https://cdn0.iconfinder.com/data/icons/cables-plugs/154/socket-electricity-cable-connect-plug-512.png"
              }
            ]
          },
          "rateable": true,
          "rating": "null",
          "price": {
            "value": "200",
            "currency": "USD",
            "discounted_value": 180.0,
            "reward_based_discount_percent": 10
          },
          "quantity": {
            "available": {
              "count": 20
            }
          },
          "category_ids": [
            "44"
          ],
          "fulfillment_ids": [
            "617"
          ],
          "location_ids": [
            "29"
          ],
          "tags": [
            {
              "descriptor": {
                "description": "connection-type",
                "code": "connection-type"
              },
              "list": [
                {
                  "value": "Commercial",
                  "descriptor": {
                    "code": "property-type",
                    "name": "property-type"
                  },
                  "display": true
                }
              ]
            },
            {
              "descriptor": {
                "description": "documentation",
                "code": "documentation"
              },
              "list": [
                {
                  "value": "Three Phase",
                  "descriptor": {
                    "code": "phase",
                    "name": "phase"
                  },
                  "display": true
                },
                {
                  "value": "Three Phase",
                  "descriptor": {
                    "code": "phase",
                    "name": "phase"
                  },
                  "display": true
                }
              ]
            }
          ]
        }
      ]
    }
  ]
}


